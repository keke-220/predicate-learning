# Predicate_Learning

## Installation
install your local pomdp solver

download dataset at https://www.eecs.tufts.edu/~gtatiya/pages/2014/CY101Dataset.html

remove no_object from sensorimotor feature files

create data directory

```
mkdir data/cy101/normalized_data_without_noobject
```

## Run
run online predicate learning simulation

```python
python learning_10pred.py 'YOUR_LOCAL_POMDP_SOL_PATH'
```
