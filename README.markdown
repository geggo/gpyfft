gpyfft
======
A Python wrapper for the OpenCL FFT library APPML/clAmdFft from AMD

Introduction
------------
AMD has created a nice FFT library for use with their OpenCL implementation
called:

AMD Accelerated Parallel Processing Math Libraries
http://developer.amd.com/libraries/appmathlibs/Pages/default.aspx

But it doesn't generate a set of kernels and hand them back to the user to queue
up with the OpenCL kernel and event management system on his own.

Its programming model wants to control that part of the transaction, handing 
control back to the user only after the results are written to the output buffers.

So, if one wants to use this handy library with one's AMD setup and get the
benefit of PyOpenCL's simplified OpenCL interface this library wrapper is needed 
to make the AMD FFT library play nicely with PyOpenCL.

Requirements
------------
* python
* cython
* pyopencl
* APPML clAmdFft 1.8
* AMD APP SDK

Installation
------------

* install clAmdFft, add clAmdFft/binXX to PATH (or copy
  clAmdFft.Runtime.dll to package directory), edit setup.py to point
  to clAmdFft and AMD APP directories.

  And either:
  
* python setup.py install

  Or:
* inplace build:
  python setup.py build_ext --inplace

License:
--------

GPL
(does it have to be a GPL?  Can we use one that's better for commercial and non-commercial use?)

Tested Platforms
----------------
================  ============================================================
Directive Name    Description (Docutils version added to, in [brackets])
================  ============================================================
attention         Specific admonition; also "caution", "danger",
                  "error", "hint", "important", "note", "tip", "warning"
admonition        Generic titled admonition: ``.. admonition:: By The Way``
image             ``.. image:: picture.png``; many options possible
figure            Like "image", but with optional caption and legend
topic             ``.. topic:: Title``; like a mini section
sidebar           ``.. sidebar:: Title``; like a mini parallel document
parsed-literal    A literal block with parsed inline markup
rubric            ``.. rubric:: Informal Heading``
epigraph          Block quote with class="epigraph"
highlights        Block quote with class="highlights"
pull-quote        Block quote with class="pull-quote"
compound          Compound paragraphs [0.3.6]
container         Generic block-level container element [0.3.10]
table             Create a titled table [0.3.1]
list-table        Create a table from a uniform two-level bullet list [0.3.8]
csv-table         Create a table from CSV data (requires Python 2.3+) [0.3.4]
contents          Generate a table of contents
sectnum           Automatically number sections, subsections, etc.
header, footer    Create document decorations [0.3.8]
target-notes      Create an explicit footnote for each external target
math              Mathematical notation (input in LaTeX format)
meta              HTML-specific metadata
include           Read an external reST file as if it were inline
raw               Non-reST data passed untouched to the Writer
replace           Replacement text for substitution definitions
unicode           Unicode character code conversion for substitution defs
date              Generates today's date; for substitution defs
class             Set a "class" attribute on the next element
role              Create a custom interpreted text role [0.3.2]
default-role      Set the default interpreted text role [0.3.10]
title             Set the metadata document title [0.3.10]
================  ============================================================


(C) Gregor Thalhammer 2012
