# Strata

Final Project for CS 229 by Jacob Thompson and Christy Jestin

# Running the Demo

Click on "visualization.py" to run the demo.
The player will be auto-attacking so that graders do not need to fiddle with controls; simply use W,A,S,D and R to move the player character and rotate it, respectively. Attempt to reduce the enemy health to 0 while preserving your own health! This represents a simple strategy by the adversary: maintain position and attack in circles as a defense.

# Understanding the Model

The main model uses submodules as abstractions for the tasks of value prediction, policy prediction/generation, state evolution, and strategy distillation (sometimes referred to as memory module). The prediction tasks also share a common submodule for state analysis. All of the top level logic is in `model.py`, and the submodules are implemented in their own files.

The primary design choices are in `model.py` as each submodule can easily be replaced with a different architecture and the top level logic of the model would still function. The model code is not in a runnable state, but the main model's forward pass is fully implemented. The backward pass is outdated and needs to be revamped to backprop on an entire game at once. The strategy distillation module has not been implemented yet, but all other submodules are complete.
