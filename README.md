# deal.II implmentation of multiscale equations

This repository contains a setup for a [deal.ii][1] implementation for a two-scale system of equations.
The target system is described in [this thesis][2], but the current status of the implementation is still a proof of concept.

## Installation for macOS

A bundled package in the form of an app is available [here][3]. Installing it provides your Mac with the deal.II library files.
Other options are available as well and should work, but the rest of this README assumes the app installation.

## Compiling deal.II programs

deal.II uses CMake for its compilation process, as does this repo. If for some reason CMake isn't present on your system, its easily installed using [Homebrew][4].
Opening the deal.II app will open a shell with preset environment variables that make compilation of your programs 'easy'. 
For some reason, this app takes quite some time to load, and its structure doesn't (by default) allow for the use of IDE's or other shells/terminals.

Luckily, we don't need to open the deal.II app to compile it's programs, since we can point CMake to the installation.

After installation of version `9.0.0`, add the following line to your CMake:

```cmake
set(deal.II_DIR /Applications/deal.II-9.0.0.app/Contents/Resources/lib/cmake/deal.II)
FIND_PACKAGE(deal.II 9.0.0 QUIET HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR})

DEAL_II_INITIALIZE_CACHED_VARIABLES()
DEAL_II_INVOKE_AUTOPILOT()
```

For a full (but not minimal) example of a deal.II CMake file, check the repo.

## Implementations present

deal.II comes with many examples, located in `/Applications/deal.II-9.0.0.app/Contents/Resources/examples`, also found in a more legible format on the [tutorial page][5].

This repo has currently two 'roughly working' implementations:

- `simple.cpp`, a working start for the target system
- `manufactured.cpp`, an multiscale elliptic-elliptic solver with convergence testing.


## Disclaimer

This is a sandbox. Comments are sparse. So are structure and responsible coding choices.Dive in at your own risk.

[1]: https://www.dealii.org/
[2]: http://urn.kb.se/resolve?urn=urn:nbn:se:kau:diva-68686
[3]: https://www.dealii.org/download.html
[4]: https://brew.sh
