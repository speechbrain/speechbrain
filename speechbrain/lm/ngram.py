NEGINFINITY = float("-inf")


class BackoffNgramLM:
    def __init__(self, ngrams, backoffs):
        # The ngrams format is best explained by an example query:
        # P( world | <s>, hello ),
        # i.e. trigram model, probability of "world" given "<s> hello", is:
        # >>> ngrams[2][("<s>", "hello")]["world"]
        #
        # On the top level, ngrams is a dict of different history lengths,
        # and each order is a dict, with contexts (tuples) as keys
        # and (log-)distributions (dicts) as values.
        #
        # Backoffs format is a little simpler:
        # On the top level, backoffs is a list of different orders,
        # and each order is a mapping (dict)
        # from backoff context to backoff (log-)weight
        if not (
            len(backoffs) == len(ngrams) or len(backoffs) == len(ngrams) - 1
        ):
            raise ValueError("Backoffs needs to be of order N or N-1")
        self.ngrams = ngrams
        self.backoffs = backoffs
        self.top_order = len(self.ngrams)

    def logprob(self, token, context=tuple()):
        # If a longer context is given than we can ever use,
        # just use less context.
        query_order = len(context) + 1
        if query_order > self.top_order:
            return self.logprob(token, context[1:])
        # Now, let's see if we have both:
        # a distribution for the query context at all
        # and if so, a probability for the token.
        # Then we'll just return that.
        if (
            context in self.ngrams[query_order]
            and token in self.ngrams[query_order][context]
        ):
            return query_order, self.ngrams[query_order][context][token]
        # If we're here, no direct probability stored for the query.
        # Missing unigram queries are a special case, the recursion will stop.
        if query_order == 1:
            return 0, NEGINFINITY  # Zeroth order for not found
        # Otherwise, we'll backoff to lower order model.
        # First, we'll get add the backoff log weight
        context_order = query_order - 1
        backoff_log_weight = self.backoffs[context_order].get(context, 0.0)
        # And then just recurse:
        order_hit, lp = self.logprob(token, context[1:])
        return order_hit, lp + backoff_log_weight
