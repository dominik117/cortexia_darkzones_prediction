BACKLOG

- Poisson loss into tree based models
- Add feature: neighborhood population
- Add feature: parks and events
- Add feature: count of meassurements per Edge per day
- Add feature: nearby edges (their cleanliness? Other features from them?)
- Add a validation set to the training
- Make a log of the tested models
- Find the feature importance
- How does the prediction perform on low occuring edges
- Aggregated sum of cigarettes per month both with and without predicted counts

- High level absolute metrics
- Try Neural Networks because dependencies are not obvious and there are many of them.
- Try ‘edge_id’ - ‘date_utc’ tables and use IterativeImputer or FancyImputer to calculate values from surrounding edges and dates.
- Use PowerTrainsformer and LabelTransformer on the Features, because all of them are not Gaussian, therefore Scaling is not enough.

