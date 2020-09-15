# Predicate_Learning

## Installation
1. install your local pomdp solver

2. download dataset at https://www.eecs.tufts.edu/~gtatiya/pages/2014/CY101Dataset.html

3. remove no_object from sensorimotor feature files

4. create a dataset directory and copy all feature files into it

```
mkdir data/cy101/normalized_data_without_noobject
```

## Run
run online predicate learning simulation

```python
python learning_10pred.py 'YOUR_LOCAL_POMDP_SOL_PATH'
```
