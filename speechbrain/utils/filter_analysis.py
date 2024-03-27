"""Implements utils to model and combine filter properties, i.e. compute how
window size, stride, etc. behave, which may be useful for certain usecases such
as streaming.

Authors:
 * Sylvain de Langen 2024
"""

from dataclasses import dataclass


@dataclass
class FilterProperties:
    """Models the properties of something that behaves like a filter (e.g.
    convolutions, fbanks, etc.) over time.
    """

    window_size: int
    """Size of the filter, i.e. the number of input frames on which a single
    output depends. Other than dilation, it is assumed that the window operates
    over a contiguous chunk of frames.

    Example:
    --------
    .. code-block:: text

        size = 3, stride = 3

        out  <-a-> <-b-> <-c->
        in   1 2 3 4 5 6 7 8 9
    """

    stride: int = 1
    """Stride of the filter, i.e. how many input frames get skipped over from an
    output frame to the next (regardless of window size or dilation).

    Example:
    --------
    .. code-block:: text

        size = 3, stride = 2

             <-a->
                 <-b->   <-d->
        out          <-c->
        in   1 2 3 4 5 6 7 8 9
    """

    dilation: int = 1
    """Dilation rate of the filter. A window will consider every n-th
    (n=dilation) input frame. With dilation, the filter will still observe
    `size` input frames, but the window will span more frames.

    Dilation is mostly relevant to "a trous" convolutions.
    A dilation rate of 1, the default, effectively performs no dilation.

    Example:
    --------
    .. code-block:: text

        size = 3, stride = 1, dilation = 3

            <-------> dilation - 1 == 2 skips
            a        a        a
            |  b     |  b     |  b
            |  |  c  |  |  c  |  |  c
            |  |  |  d  |  |  d  |  |  d
            |  |  |  |  e  |  |  e  |  |  ..
        in  1  2  3  4  5  6  7  8  9  10 ..
            <-> stride == 1
    """

    causal: bool = False
    """Whether the filter is causal, i.e. whether an output frame only depends
    on past input frames (of a lower or equal index).

    In certain cases, such as 1D convolutions, this can simply be achieved by
    inserting padding to the left of the filter prior to applying the filter to
    the input tensor.

    Example:
    --------
    .. code-block:: text

        size = 3, stride = 1, causal = true
                 <-e->
               <-d->
             <-c->
             b->
             a
        in   1 2 3 4 5
    """

    def __post_init__(self):
        assert self.window_size > 0
        assert self.stride > 0
        assert (
            self.dilation > 0
        ), "Dilation must be >0. NOTE: a dilation of 1 means no dilation."

    @staticmethod
    def pointwise_filter() -> "FilterProperties":
        """Returns filter properties for a trivial filter whose output frames
        only ever depend on their respective input frame.
        """
        return FilterProperties(window_size=1, stride=1)

    def get_effective_size(self):
        """The number of input frames that span the window, including those
        ignored by dilation.
        """
        return 1 + ((self.window_size - 1) * self.dilation)

    def get_convolution_padding(self):
        """The number of frames that need to be inserted on each end for a
        typical convolution.
        """
        if self.window_size % 2 == 0:
            raise ValueError("Cannot determine padding with even window size")

        if self.causal:
            return self.get_effective_size() - 1

        return (self.get_effective_size() - 1) // 2

    def get_noncausal_equivalent(self):
        """From a causal filter definition, gets a compatible non-causal filter
        definition for which each output frame depends on the same input frames,
        plus some false dependencies.
        """
        if not self.causal:
            return self

        return FilterProperties(
            # NOTE: valid even on even window sizes e.g. (2-1)*2+1 == 3
            window_size=(self.window_size - 1) * 2 + 1,
            stride=self.stride,
            dilation=self.dilation,
            causal=False,
        )

    def with_on_top(self, other, allow_approximate=True):
        """Considering the chain of filters `other(self(x))`, returns
        recalculated properties of the resulting filter.

        Arguments
        ---------
        other: FilterProperties
            The filter to combine `self` with.

        allow_approximate: bool, optional
            If `True` (the default), the resulting properties may be
            "pessimistic" and express false dependencies instead of erroring
            out when exact properties cannot be determined.
            This might be the case when stacking non-causal and causal filters.
            Depending on the usecase, this might be fine, but functions like
            `has_overlap` may erroneously start returning `True`.

        Returns
        -------
        FilterProperties
            The properties of the combined filters.
        """
        self_size = self.window_size

        if other.window_size % 2 == 0:
            if allow_approximate:
                other_size = other.window_size + 1
            else:
                raise ValueError(
                    "The filter to append cannot have an uneven window size. "
                    "Specify `allow_approximate=True` if you do not need to "
                    "analyze exact dependencies."
                )
        else:
            other_size = other.window_size

        if (self.causal or other.causal) and not (self.causal and other.causal):
            if allow_approximate:
                return self.get_noncausal_equivalent().with_on_top(
                    other.get_noncausal_equivalent()
                )
            else:
                raise ValueError(
                    "Cannot express exact properties of causal and non-causal "
                    "filters. "
                    "Specify `allow_approximate=True` if you do not need to "
                    "analyze exact dependencies."
                )

        out_size = self_size + (self.stride * (other_size - 1))
        stride = self.stride * other.stride
        dilation = self.dilation * other.dilation
        causal = self.causal

        return FilterProperties(out_size, stride, dilation, causal)


def stack_filter_properties(filters, allow_approximate=True):
    """Returns the filter properties of a sequence of stacked filters.
    If the sequence is empty, then a no-op filter is returned (with a size and
    stride of 1).

    Arguments
    ---------
    filters: FilterProperties | any
        The filters to combine, e.g. `[a, b, c]` modelling `c(b(a(x)))`.
        If an item is not an instance of :class:`FilterProperties`, then this
        attempts to call `.get_filter_properties()` over it.
    allow_approximate: bool, optional
        See `FilterProperties.with_on_top`.

    Returns
    -------
    ret: FilterProperties
        The properties of the sequence of filters
    """
    ret = FilterProperties.pointwise_filter()

    for prop in filters:
        if not isinstance(prop, FilterProperties):
            prop = prop.get_filter_properties()

        ret = ret.with_on_top(prop, allow_approximate)

    return ret
