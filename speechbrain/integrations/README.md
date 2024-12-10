Integrations
------------

This python module serves to collect all the (non-recipe) SpeechBrain code that relies on
external libraries. This makes it easier to keep track of what integrations have been
added and apply different rules to the adding and maintenance of new integrations.

> [!WARNING]
> Since these integrations rely on libraries not part of the core toolkit, we make
> no guarantees as to the proper functioning of these libraries and they may be
> broken at any point in time.

In order to minimize the impact of libraries changing and causing the integrations
to stop functioning, we will add additional tests and checks on code in this module.
If the tests are broken, we may remove rather than fix the code in this integration
depending on our capacity.
