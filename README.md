# Freeze Casting - Porosity Prediction
## Database
The data used in this project is obtained from freezecasting.net : http://www.freezecasting.net/downloads.html


The schema xml file was downloaded and loaded with phpmyAdmin in a mysql database.


After loading the schema in to MySQL and connectin to python. The features of interest are loaded into a pandas dataframe, following the sql queries from the notebook in
http://www.freezecasting.net/downloads.html

The dataframe was filtered and saved as csv in `data` folder.

## Training
Training examples and results are displayed in train.ipynb.


Trained models are saved in `temp_models` folder.


Train pipeline configuration is defined at custom_models.py


Grid search params are defined in `grid_search_params.py`


