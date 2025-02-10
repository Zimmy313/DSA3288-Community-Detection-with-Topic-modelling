# Changelog
All notable changes since 2024-02-07 to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Added
- 

### Changed
- hLDA does not count root as one of the total table generated. Set total_created_nodes = 1 during initialisation of hLDA.
- gibbs_sampling now return the time taken to complete.
- gibbs_sampling now takes in an arguement to control printing of structure.

### Fixed
-

### Things to be changed
- Generate synthetic data.
- Analyse the hyperparameter test base on what i have now. Then, we can try a few more models(with different seed for initialisation) and take average.

## [1.0.0] - 2024-02-07
### Added
- 