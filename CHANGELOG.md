# Changelog

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.2.5](https://crates.io/crates/sized_matrix/0.2.5) - 2021-04-05

### Fixed:
* Typo in `Vector` `IndexMut` causing index out of bounds error

## [v0.2.4](https://crates.io/crates/sized_matrix/0.2.4) - 2021-04-04

### Fixed:
* Unstable function name changed by <https://github.com/rust-lang/rust/issues/63567>

## [v0.2.3](https://crates.io/crates/sized_matrix/0.2.3) - 2020-11-12

### Added:
* Added left-division for `Matrix` (`rhs^-1 * self` instead of `self * rhs^-1`)

## [v0.2.2](https://crates.io/crates/sized_matrix/0.2.2) - 2020-09-08

### Added:
* Implemented `Zip` and `Section` from [higher_order_functions](https://crates.io/crates/higher_order_functions) for `Vector` and `Matrix`

## [v0.2.1](https://crates.io/crates/sized_matrix/0.2.1) - 2020-09-06

### Added:
* Dot product for vectors

## [v0.2.0](https://crates.io/crates/sized_matrix/0.2.0) - 2020-09-04

### Changed:
* Moved from [init_trait](https://crates.io/crates/init_trait) to [higher_order_functions](https://crates.io/crates/higher_order_functions)

### Added:
* Implemented [Map](https://docs.rs/higher_order_functions/0.1.0/higher_order_functions/trait.Map.html) for matrices

## [v0.1.0](https://crates.io/crates/sized_matrix/0.1.0) - 2020-08-23

### Added:
* Initial implementation
