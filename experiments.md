Final set of cols to use:
```
Class,C1,C3,C4,C5,C6,C7,C8,C9,C13,C14,C16,C18,C19,C20,C21,C22,C23,C24,C27,C28,C29
```

<!-- ## EX. 1: Scaling Type - CANCELLED
- Use defined base hparams for each model
- All features
- No sampling
- Switch between scaling types
- No binning (obviously)
- All models -->

## Ex. 1: Binnning Vs No Binning
- Use defined base hparams for each model
- All features
- No sampling
- One with binning to 10 bins, one with no binning
- All models

## EX. 2: Feature Selection
- Use defined base hparams for each model
- Drop one feature at a time
- No sampling
- Best binning (binning or no binning) from 1
- All models

## EX. 3: Sampling
- Use defined base hparams for each model
- Switch between sampling methods
- Use dropped feature set decided from 3
- Best binning (binning or no binning) from 2
- All models

## EX. 4: Hyperparameter Tuning
- Use big set of hparams to tune properly for each model
- Use best sampling method for each model from 3
- Use dropped feature set decided from 3
- Best binning (binning or no binning) from 2
- All models