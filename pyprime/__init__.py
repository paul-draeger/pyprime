"""PyPrime package root."""

__version__ = "1.0.0"

# Expose frequently used functions and classes on the package namespace.
# This lets you do:
#   import pyprime as pp
#   pp.sim(...)
#   pp.analyse_Re(...)
#
# If you prefer a more structured import, use the subpackages:
#   pp.analysis.sim(...)
#   pp.misc.download(...)

from .analysis.sim import *
