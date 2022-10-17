# gru-valuation

This project uses a PyTorch to create a GRU-based recurrent neural network that, given a company's financials for a period of years, outputs a prediction of the company's market cap—in essence, a model to value a company given its financials.

## Architecture

A simple, three-layer conventional RNN architecture is used: an input layer, taking a pre-processed (discussed in a moment) selection of a company's financial data across some number of years; a gated recurrent unit (GRU) layer; and a fully-connected, linear output layer.

## Pre-Processing and Scaling

As financial data often exists in large numbers, the need to perform pre- and post-processing steps to scale values up or down arises. Thus, we'll use a variation of exponential scaling to shrink the domain of our data prior to feeding it to the model:

    def scale_down(x: torch.Tensor):
        return (x.abs() + 1).log() * x.sign()

This function, when graphed, looks like a horizontally-stretched "S" shape, and is defined for all real numbers:

<img src="scale_down_graph.png" alt="'S' shaped graph" width="400"/>

The inverse of this function—to scale small outputs up—is applied to the outputs of the network, too.

## Data Sourcing

Good quality historical financial data is hard to find. So far, the most complete, reasonably priced dataset I've encountered is the [Sharadar Core US Fundamentals](https://data.nasdaq.com/databases/SF1/data) on Nasdaq Data Link. It provides a litany of metrics for the past 20-ish years across 16,000 companies.

While using data from any source should be possible, I've written an "importer" for the Sharadar database. This takes the CSV from Nasdaq Data Link, and moves it to a SQLite database for easier storage and retrieval.
