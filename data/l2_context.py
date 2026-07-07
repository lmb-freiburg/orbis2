class L2ContextMixin:
    """Reusable helper for hierarchical L1/L2 frame alignment."""

    def _init_l2_context(
        self,
        *,
        num_l2_context=0,
        l2_frame_rate=1.0,
        l1_context_frames=1,
        require_l2_context=False,
    ):
        self.num_l2_context = int(num_l2_context)
        self.l1_context_frames = int(l1_context_frames)

        if self.num_l2_context < 0:
            raise ValueError(f"num_l2_context must be non-negative, got {self.num_l2_context}.")
        if self.l1_context_frames <= 0:
            raise ValueError(f"l1_context_frames must be positive, got {self.l1_context_frames}.")
        if require_l2_context and self.num_l2_context <= 0:
            raise ValueError("num_l2_context must be positive for hierarchical L1/L2 loaders.")

        self.l2_frame_interval = None
        self._l2_anchor_offset = (self.l1_context_frames - 1) * self.frame_interval
        if self.num_l2_context == 0:
            return

        ratio = float(self.stored_data_frame_rate) / float(l2_frame_rate)
        rounded_ratio = round(ratio)
        if abs(ratio - rounded_ratio) > 1e-8:
            raise ValueError(
                "stored_data_frame_rate must be an integer multiple of l2_frame_rate, "
                f"got stored_data_frame_rate={self.stored_data_frame_rate}, "
                f"l2_frame_rate={l2_frame_rate}."
            )
        self.l2_frame_interval = int(rounded_ratio)

    @property
    def l2_context_enabled(self):
        return self.num_l2_context > 0

    def get_required_l1_start_offset(self):
        if not self.l2_context_enabled:
            return 0
        return max(
            0,
            (self.num_l2_context - 1) * self.l2_frame_interval - self._l2_anchor_offset,
        )

    def has_l2_context_for_start(self, start_frame):
        if not self.l2_context_enabled:
            return True
        l2_anchor = start_frame + self._l2_anchor_offset
        oldest_l2 = l2_anchor - self.l2_frame_interval * (self.num_l2_context - 1)
        return oldest_l2 >= 0

    def filter_index_map_with_l2_headroom(self, index_map, start_frame_idx=-1):
        if not self.l2_context_enabled:
            return list(index_map)

        filtered = []
        for item in index_map:
            start_frame = item[start_frame_idx]
            if self.has_l2_context_for_start(start_frame):
                filtered.append(item)
        return filtered

    def get_l1_indices(self, start_frame, num_frames):
        return list(range(
            start_frame,
            start_frame + num_frames * self.frame_interval,
            self.frame_interval,
        ))

    def get_l2_indices(self, start_frame):
        if not self.l2_context_enabled:
            return []

        l2_end = start_frame + self._l2_anchor_offset
        l2_start = l2_end - (self.num_l2_context - 1) * self.l2_frame_interval
        return list(range(l2_start, l2_end + 1, self.l2_frame_interval))

    def get_l1_and_l2_indices(self, start_frame, num_frames):
        return self.get_l1_indices(start_frame, num_frames), self.get_l2_indices(start_frame)
