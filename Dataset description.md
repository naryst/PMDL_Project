This dataset contains code changes in each commit of most starred python project, stored on GitHub. 

## Code to reproduce the parsing process
To parse code we performed the following steps:
* Get list of most starred GitHub repos via API
* With **git** python package clone all the repos from the list to local machine and write code defference for each commit of every repo to the dataset.
* Clean dataset to remove to large commits, commits with not python code changes, commits with non-ASCII chars, etc.
* Group files changed in 1 commit into single sample of the dataset.
To reproduce these steps you need to:
1) run *src/github_parsing.ipynb* to parse repos from github  
2) to clean the data and group dataset samples run *src/data_cleaning.ipynb*  

## Dataset features
Dataset have the following features:
1) repo_name
2) commit_message
3) commit_changes - changes in code in all python files, contained in the commit
4) files_changed - number of files, changed in the commit
5) changes_len - number of chars in the code changes
For model training we used only *commit_message* feature as a label and *commit_changes* as an input for the model.
Code changes have the following structure:
```
<filename> name_of_the_file <filename>
code_of_changes
<commit_msg>
```
Special tokens used in the input:
* <file_name> - used to separate name of the file
* <code_del> and <code_add> used to separate added or deleted lines of code in the commit
* <commit_msg> used to separate commit message 

Example of input for the model:
```
<filename> a/tests/test_constraint.py b/tests/test_constraint.py<filename>
<code_del>--- a/tests/test_constraint.py<code_del>
<code_add>+++ b/tests/test_constraint.py<code_add>
@@ -87,10 +87,15 @@ def test_accurate_approximation_when_known():
         n_iter=10,
     )
 
<code_del>-    params = optimizer.res[0]["params"]<code_del>
<code_del>-    x, y = params['x'], params['y']<code_del>
<code_add>+    # Exclude the last sampled point, because the constraint is not fitted on that.<code_add>
<code_add>+    res = np.array([[r['target'], r['constraint'], r['params']['x'], r['params']['y']] for r in optimizer.res[:-1]])<code_add>
<code_add>+<code_add>
<code_add>+    xy = res[:, [2, 3]]<code_add>
<code_add>+    x = res[:, 2]<code_add>
<code_add>+    y = res[:, 3]<code_add>
     
<code_del>-    assert constraint_function(x, y) == approx(conmod.approx(np.array([x, y])), rel=1e-5, abs=1e-5)<code_del>
<code_add>+    assert constraint_function(x, y) == approx(conmod.approx(xy), rel=1e-5, abs=1e-5)<code_add>
<code_add>+    assert constraint_function(x, y) == approx(optimizer.space.constraint_values[:-1], rel=1e-5, abs=1e-5)<code_add>
 
 
 def test_multiple_constraints():

<commit_msg>In case of commit with the several files changed, different files are separated with 3 blank lines.<eos>


```
In case of commit with the several files changed, different files are separated with 3 blank lines.