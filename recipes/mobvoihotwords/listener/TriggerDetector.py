class TriggerDetector:
    """
    Reads predictions and detects activations
    This prevents multiple close activations from occurring when
    the predictions look like ...!!!..!!...
    """
    def __init__(self, chunk_size, sensitivity=0.5, trigger_level=3):
        self.chunk_size = chunk_size
        self.sensitivity = sensitivity
        self.trigger_level = trigger_level
        self.activation = 0
        self.prob_buffer = [0]*9

    def update(self, prob):
        # type: (float) -> bool
        """Returns whether the new prediction caused an activation"""
        chunk_activated = prob > 1.0 - self.sensitivity

        if chunk_activated or self.activation < 0:
            self.activation += 1
            has_activated = self.activation > self.trigger_level
            if has_activated or chunk_activated and self.activation < 0:
                self.activation = -(8 * 2048) // self.chunk_size

            if has_activated:
                return True
        elif self.activation > 0:
            self.activation -= 1
        return False

    def med_filter(self, prob):
        # type: (float) -> float
        """Returns whether the new prediction caused an activation"""
        self.prob_buffer[0:-1] = self.prob_buffer[1:]
        self.prob_buffer[-1] = prob
        prob_sorted = sorted(self.prob_buffer)

        return prob_sorted[int((len(self.prob_buffer)-1)/2)]