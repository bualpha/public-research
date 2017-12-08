# research-papers

A collection of research papers that are useful for our own internal research and for implementations made in the summer program.

## Where to Find Papers?

There are many resources for finding trading strategy papers, though we use mainly the following:

  - [SSRN](https://papers.ssrn.com/sol3/DisplayAbstractSearch.cfm)
  - [arXiv](https://arxiv.org/)
  - [Quantpedia](https://quantpedia.com/Screener)

## Submitting Papers with Implementations

You can clone the repository by running the following in your terminal:

`git clone git@github.com:bualpha/research-papers.git`

Then add a link to the paper with its name and the abstract of the paper, all formatted in the same way as the existing papers in the README, along with your Jupyter Notebook and code into their respective folders.

If you make any changes to the repo, please use the following acronyms to prefix your commit messages:

```
BUG: bug fix
DEP: deprecate something, or remove a deprecated object
DEV: development tool or utility
DOC: documentation
ENH: enhancement
MAINT: maintenance commit (refactoring, typos, etc.)
REV: revert an earlier commit
STY: style fix (whitespace, PEP8)
TST: addition or modification of tests
```

**List of Papers w/ Abstracts**

  - [Momentum Has Its Moments](http://docentes.fe.unl.pt/~psc/MomentumMoments.pdf)
  ```
  Compared with the market, value, or size factors, momentum has offered investors the
  highest Sharpe ratio. However, momentum has also had the worst crashes, making the
  strategy unappealing to investors who dislike negative skewness and kurtosis. We find
  that the risk of momentum is highly variable over time and predictable. Managing this
  risk virtually eliminates crashes and nearly doubles the Sharpe ratio of the momentum
  strategy. Risk-managed momentum is a much greater puzzle than the original version.
  ```
  - [130-30 The New Long Only](https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf)
  ```
  One of the fastest growing areas in institutional investment management is the so-called active extension or 130/30
  class of strategies in which the short-sales constraint of a traditional long-only portfolio is relaxed. Fueled both by
  the historical success of long-short equity hedge funds and the increasing frustration of portfolio managers at the
  apparent impact of long-only constraints on performance, 130/30 products have grown to over $75 billion in assets and
  could reach $2 trillion by 2010 (Tabb and Johnson [2007]).
  
  Despite the increasing popularity of such strategies, considerable confusion still exists among managers and investors
  regarding the appropriate risks and expected returns of 130/30 products. For example, the typical 130/30 portfolio has a
  leverage ratio of 1.6 to 1, unlike a long-only portfolio that does not use leverage. Although leverage is typically
  associated with higher volatility returns, the volatility and market beta of a typical 130/30 portfolio are
  comparable to those of its long-only counterpart. Nevertheless, the added leverage of a 130/30 product
  suggests that the expected return should be higher than its long-only counterpart, but by how much? By definition,
  a 130/30 portfolio holds 130% of its capital in long positions and 30% in short positions. Thus,
  the 130/30 portfolio may be viewed as a long-only portfolio plus a market-neutral portfolio with long and short
  exposures that are 30% of the long-only portfolio’s market value. The active portion of a 130/30 strategy,
  however, is typically very different from a market- neutral portfolio so that this
  decomposition is, in fact, inappropriate.
  
  These unique characteristics suggest that existing indexes such as the S&P 500 and the Russell 1000 are inappropriate
  benchmarks for leveraged dynamic portfolios such as 130/30 funds. A new benchmark is needed, one that incorporates the same
  leverage constraints and portfolio construction algorithms as 130/30 funds, but is otherwise transparent, investable, and
  passive. We provide such a benchmark in this article.
  ```
      
# License

[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)