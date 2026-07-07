import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl

from omegaconf import ListConfig, DictConfig

import logging

from util import instantiate_from_config

logger = logging.getLogger(__name__)


def _collate_pad_missing(batch):
    """Collate dicts that may have different keys across datasets.

    Missing keys are filled with zero tensors matching the shape of the first
    sample in the batch that has that key. Non-tensor values are filled with
    None. Allows heterogeneous datasets (e.g. with/without steering) to be
    mixed in the same batch.
    """
    if not isinstance(batch[0], dict):
        return default_collate(batch)

    all_keys = set().union(*[item.keys() for item in batch])
    filled = []
    for item in batch:
        item = dict(item)
        for key in all_keys:
            if key not in item:
                ref = next((b[key] for b in batch if key in b), None)
                if isinstance(ref, torch.Tensor):
                    item[key] = torch.full_like(ref, float("nan"))
                else:
                    item[key] = ref
        filled.append(item)
    return default_collate(filled)


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning("Invalid boolean %s=%r; using default %s", name, raw, default)
    return default


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer %s=%r; using default %s", name, raw, default)
        return default


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size=None, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, dbg=False, train_weights=None):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.dbg = dbg

        if train_weights is not None:
            if not isinstance(train, (list, ListConfig)):
                raise ValueError("train_weights requires train to be a list of dataset configs")
            if len(train_weights) != len(train):
                raise ValueError(
                    f"train_weights has {len(train_weights)} entries but train has {len(train)} datasets"
                )
        self.train_weights = train_weights

        if self.wrap:
            raise NotImplementedError("Wrapped datasets not implemented")

        self._resume_epoch = None
        self._resume_batches_completed = 0

    def state_dict(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return {}

        epoch = int(getattr(trainer, "current_epoch", 0))
        batches_completed = 0
        try:
            # Lightning tracks completed batches for the current epoch in this counter.
            batches_completed = int(trainer.fit_loop.epoch_loop.batch_progress.current.completed)
        except Exception:
            batches_completed = 0

        return {
            "resume_epoch": epoch,
            "resume_batches_completed": max(batches_completed, 0),
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return

        self._resume_epoch = state_dict.get("resume_epoch")
        self._resume_batches_completed = int(state_dict.get("resume_batches_completed", 0) or 0)
        if self._resume_batches_completed < 0:
            self._resume_batches_completed = 0

    def setup(self, stage=None):
        self.datasets = dict()
        for k, cfg in self.dataset_configs.items():
            logger.info("Loading dataset: %s", k)
            if isinstance(cfg, (list, ListConfig)):
                datasets = [instantiate_from_config(c) for c in cfg]
                self.datasets[k] = ConcatDataset(datasets)
                [logger.info(d) for d in datasets]
            elif isinstance(cfg, DictConfig):
                ds = instantiate_from_config(cfg)
                self.datasets[k] = ds
                logger.info(ds)
            else:
                raise ValueError(f"Invalid dataset config: {cfg}")

    def _train_dataloader(self):
        is_distributed = dist.is_available() and dist.is_initialized()
        use_distributed_sampler = _env_bool("ORBIS_USE_DISTRIBUTED_SAMPLER", True)
        sampler = None
        if self.train_weights is not None:
            train_ds = self.datasets["train"]
            sub_datasets = getattr(train_ds, "datasets", [train_ds])
            dataset_sizes = [len(ds) for ds in sub_datasets]
            # Anchor epoch length to min(size/weight) across datasets.
            # This fully covers the most "weight-adjusted-constrained" dataset (typically
            # the highest-weight one) with exactly one pass, while letting smaller/lower-weight
            # datasets cycle. Avoids the 2x repetition that len(ConcatDataset) causes when
            # one dataset is large and dominates with high weight.
            num_samples = math.ceil(min(s / w for s, w in zip(dataset_sizes, self.train_weights)))
            sampler = _WeightedDistributedSampler(
                self.train_weights, dataset_sizes, num_samples,
                num_replicas=dist.get_world_size() if (is_distributed and use_distributed_sampler) else 1,
                rank=dist.get_rank() if (is_distributed and use_distributed_sampler) else 0,
            )
            logger.info(
                "Train DataLoader: weighted sampling fractions=%s dataset sizes=%s num_samples/epoch=%s",
                list(self.train_weights),
                dataset_sizes,
                num_samples,
            )
        elif is_distributed and use_distributed_sampler:
            sampler = _ResumableDistributedSampler(
                self.datasets["train"],
                shuffle=True,
                drop_last=True,
                resume_epoch=self._resume_epoch,
                resume_batches_completed=self._resume_batches_completed,
                batch_size=self.batch_size,
            )

        timeout_s = _env_int("ORBIS_DATALOADER_TIMEOUT_S", 0)
        prefetch_factor = _env_int("ORBIS_DATALOADER_PREFETCH_FACTOR", 1)
        persistent_workers = _env_bool(
            "ORBIS_DATALOADER_PERSISTENT_WORKERS",
            self.num_workers > 0,
        )
        mp_context = os.environ.get("ORBIS_DATALOADER_MP_CONTEXT", "").strip()

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
            timeout=max(timeout_s, 0),
            collate_fn=_collate_pad_missing if self.train_weights is not None else None,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor > 0:
                loader_kwargs["prefetch_factor"] = prefetch_factor
            if mp_context:
                loader_kwargs["multiprocessing_context"] = mp_context

        logger.info(
            "Train DataLoader: distributed=%s sampler=%s workers=%s timeout_s=%s mp_context=%s",
            is_distributed,
            sampler.__class__.__name__ if sampler is not None else "None",
            self.num_workers,
            loader_kwargs["timeout"],
            mp_context or "<default>",
        )

        if self.dbg:
            dbg_sampler = DistributedSampler(self.datasets["train"], shuffle=True)
            loader_kwargs["sampler"] = dbg_sampler
            loader_kwargs["shuffle"] = False

        return DataLoader(self.datasets["train"], **loader_kwargs)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.val_batch_size,
                          num_workers=self.num_workers)


class _ResumableDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        *,
        resume_epoch=None,
        resume_batches_completed=0,
        batch_size=1,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self.resume_epoch = None if resume_epoch is None else int(resume_epoch)
        self.resume_batches_completed = int(resume_batches_completed or 0)
        self.batch_size = max(int(batch_size), 1)

    def __iter__(self):
        indices = list(super().__iter__())
        if (
            self.resume_epoch is not None
            and int(self.epoch) == self.resume_epoch
            and self.resume_batches_completed > 0
        ):
            skip = self.resume_batches_completed * self.batch_size
            if skip > 0:
                logger.info(
                    "Resuming dataloader at epoch=%s after %s batches (%s samples/rank).",
                    self.resume_epoch,
                    self.resume_batches_completed,
                    skip,
                )
                indices = indices[skip:]
        return iter(indices)


class _WeightedDistributedSampler(torch.utils.data.Sampler):
    """Weighted sampler for dataset balancing, with optional distributed sharding.

    Two-stage sampling: (1) pick a dataset by weight, (2) pick a uniform index
    within that dataset. Avoids torch.multinomial's 2^24 category limit since
    only num_datasets categories are ever sampled, not num_total_samples.
    """

    def __init__(self, dataset_weights, dataset_sizes, num_samples, num_replicas=1, rank=0):
        self.dataset_weights = torch.as_tensor(dataset_weights, dtype=torch.float64)
        self.dataset_sizes = torch.tensor(dataset_sizes, dtype=torch.long)
        offsets = [0]
        for s in dataset_sizes[:-1]:
            offsets.append(offsets[-1] + s)
        self.dataset_offsets = torch.tensor(offsets, dtype=torch.long)
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples_per_replica = math.ceil(num_samples / num_replicas)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        total = self.num_samples_per_replica * self.num_replicas

        # Stage 1: pick dataset for each draw (num_datasets categories, well within 2^24)
        ds_idx = torch.multinomial(self.dataset_weights, total, replacement=True, generator=g)

        # Stage 2: pick a uniform index within each chosen dataset
        local = (torch.rand(total, generator=g) * self.dataset_sizes[ds_idx].float()).long()
        indices = (self.dataset_offsets[ds_idx] + local).tolist()

        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        return self.num_samples_per_replica