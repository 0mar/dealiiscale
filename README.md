# deal.II implementation of multiscale PDE

This repository contains a setup for a [deal.II][1] implementation for a two-scale system of equations.
The target system is described in [this thesis][2], but the current status of the implementation is still a proof of concept.
A more detailed description of the mathematical framework behind this implementation is present in `latex/main.tex`.

## Requirements

 - `cmake`, preferable version 3.1, but lower versions should work too
 - deal.II, version 9.0. See below for installation hints

## Installation for macOS

A bundled package in the form of an app is available [here][3]. Installing it provides your Mac with the deal.II library files.
Other options are available as well and should work, but the rest of this README assumes the app installation.

## Installation for Linux (Ubuntu)

Most popular distros have deal.II available in their repos. For Ubuntu, deal.II is installed using

```bash
sudo apt install lib-deal.ii-dev
``` 

*Caveat*: On the latest Ubuntu LTS (18.04), at the time of writing, the deal.II version available is 8.5.
Unfortunately, some of the (even basic) tutorial programs are not backwards compatible with this version.
In order to install deal.II 9.0.0, either compile from source or add the `universe` component of the 18.10 edition of Ubuntu to `apt`:

```bash
echo 'deb http://archive.ubuntu.com/ubuntu/ cosmic main universe' | sudo tee -a /etc/apt/sources.list
sudo apt update
```
But be aware that this last option will update all of software from the repository to its 18.10 version, which in some cases might not be recommendable.

## Configuring CMake for deal.II

deal.II uses CMake for its compilation process, as does this repo. If for some reason CMake is not present on your system, it is easily installed using [Homebrew][4] (Mac).
Opening the deal.II app will open a shell with preset environment variables that make compilation of your programs 'easy'. 
For some reason, this app takes quite some time to load, and its structure does not (by default) allow for the use of IDEs or other shells/terminals.

Luckily, we do not need to open the deal.II app to compile our programs, since we can point CMake to the installation.

After installation of version `9.0.0`, add the following lines to your CMake:

```cmake
set(deal.II_DIR /Applications/deal.II-9.0.0.app/Contents/Resources/lib/cmake/deal.II)
FIND_PACKAGE(deal.II 9.0.0 QUIET HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
```

For default installations on Linux systems, the first line can be changed to
```cmake
set(deal.II_DIR /usr/share/cmake/deal.II)
```
For a full (but not minimal) example of a deal.II CMake file, check the repository. This CMake file deals with both Mac and Ubuntu installations.

## Compiling the source code

To build and install the binaries in directory `bin` with a functioning CMake file, one can run the following commands:

```bash
git clone git@github.com:0mar/dealiiscale.git # if you haven't cloned the repository yet
mkdir bin
cd bin
cmake ..
make
``` 

This builds a debug version of the code. One can then run any of the created binaries, e.g. `manufactured`, like this:

```bash
cd bin
./manufactured
``` 

## Solver gallery

deal.II comes with many examples, located in `/Applications/deal.II-9.0.0.app/Contents/Resources/examples`, also found in a more legible format on the [tutorial page][5].

This repo has a number of working implementations, most of which are used to test the performance of certain deal.II characteristics.
Currently, the working implementations are:

- `elliptic-elliptic`
    * `solve_elliptic`, a multiscale elliptic-elliptic solver with convergence testing.
    * `test_elliptic_separate` decoupled elliptic-elliptic solver used to test individual convergence properties of the coupled elliptic-elliptic implementation.
- `elliptic-parabolic`
    * `solve_parabolic`, a multiscale elliptic-parabolic solver (work in progress, main target of this exercise).
    * `test_rho_separate`, a decoupled microscopic-only solver testing the properties of the `rho` equation solver.
- `playground` (collection of separate tests and implementations)
    * `demo`, a verbatim copy of a tutorial (step 3) to check if the installation works.
    * `dirichlet`, a Poisson problem in a circular domain. 
    * `robin`, a Poisson problem with Robin boundary conditions.
    * `simple`, a (deprecated) working start for the target system.
    * `integration`, an example/convergence test on how to compute bulk and flux integrals.


## Report

To test and create a report for the `elliptic-elliptic` implementations (currently `manufactured` and `separated`), one can do the following:
```bash
./manufactured
./separated
cd results
make
``` 

This will post-process the output of the binaries into the latex report and compile it (requires `pdftex`)

## Disclaimer

This is a work in progress. Structure and implementations can change and break at any time.
However, effort will be taken to ensure the `master` branch always contains a working version of the software.

[1]: https://www.dealii.org/
[2]: http://urn.kb.se/resolve?urn=urn:nbn:se:kau:diva-68686
[3]: https://www.dealii.org/download.html
[4]: https://brew.sh
[5]: https://www.dealii.org/developer/doxygen/deal.II/Tutorial.html
