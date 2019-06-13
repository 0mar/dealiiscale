Current structure:

 - Start a macrosolver
 - Macrosolver creates a microsolver.
   - This microsolver is setup, the boundary is set by some value
and.
 - We repeat
    - Running macro
    - Running micro

 - We output the results.


New structure

 - Manager class:
    - Has one micro class and one macro class
    - Micro class can be set with a macro solution
    - Macro class can be set with a micro solution

    - output can be given from each class
  
    - 
