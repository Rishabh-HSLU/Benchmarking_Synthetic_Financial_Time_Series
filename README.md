# **Project Description**

The increasing use of synthetic data in quantitative finance promises to revolutionize how trading
strategies are developed and tested. Synthetic datasets allow financial analysts and researchers to
evaluate algorithms under a wide range of market scenarios. However, despite rapid progress in
generative modeling, there is currently no standardized benchmark to assess the quality and
reliability of synthetic financial time series data for quantitative trading applications.
This absence of evaluation standards hinders objective comparison between generation methods
and limits the confidence of practitioners in adopting synthetic data for real-world trading research.
To address this, the project aims to design and prototype a benchmarking framework for evaluating
the quality of synthetic financial data. The benchmark will focus on criteria such as fidelity (how
closely synthetic data resembles real market dynamics), diversity (how broadly it represents possible
scenarios), and utility (its effectiveness in supporting trading strategy development).

## **Research Questions**

Exploring following research questions:

• Formalization of quality criteria: What are the essential characteristics that define “high-
quality” synthetic data for quantitative trading? (e.g., ability to replicate trade opportunities,
market structure, and risk factors)

• Survey of existing metrics: Which existing evaluation metrics for synthetic time series could
be adapted to the financial domain?

• Adaptation or design of new metrics: How can current metrics be modified, or new ones
developed to assess the specific requirements of quantitative trading datasets?

• Prototyping a benchmarking pipeline: How can a reproducible pipeline be implemented to
automatically compare different synthetic data generation approaches according to the
defined metrics?