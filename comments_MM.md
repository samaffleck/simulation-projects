Comments on "LMO_MC_efficient.py", 28-04-21
___________________________________________

- I can't spot any obvious errors in execution, but there are inefficient parts.
- Please look into the "timeit" module - this will allow you to more systematically check which blocks of code are causing the bottlenecks. You can truncate to look at e.g. the top 5% most time intensive parts of the code, then focus on improving those.
- Since you observe a small systematic shift in the data - the thought occurs that you may have an "off by one" error in evaluating one of the arrays. You might want to try changing the system size to see if this behaviour changes.

- Line 212, self.latticeu:
_________________________

- I believe "self.latticeu" is being called every time in the thermal averaging procedure. If I'm not mistaken, this is still evaluating the internal energy of the entire lattice in a triple-nested for loop. This should not be necessary - evaluate energy changes to save on overhead

- Lines 263.264
_______________

I believe these quantities will remain the same - you can probably initialise them in __init__ and remove these lines here, for a minor speed boost.

-Line 286, and thermal averaging:
________________________________

- you can do mean_nn - mean_n * mean_n, for a minor speed boost
- generally - not clear here why you partly use a Pandas dataframe, then return var and cov in a function. Why not store these in the dataframe as well? We want all the outputs recorded in a consistent format
- We will eventually want the average sublattice occupancies and not just the values at the end. The way you evaluate these are inefficient (partly because of the array structure). I suggest we overhaul this part once we move onto hard carbon, but leave as is for now.

- Line 323:
___________

- you check the time to write data to a few numpy arrays - this will be negligible for one iteration. I think you can remove this line - you instead need to focus on the functions that comprise "monte_carlo" and "thermal_averaging" via timetit - these are the parts called repeatedly

