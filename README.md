# VIAsegura

## Project Description

VIAsegura is an API that helps to use artificial intelligence models developed by the Inter-American Development Bank to automatically tag items on the streets. The tags it places are some of those needed to implement the iRAP road safety methodology. 

To use it you must contact the Inter-American Development Bank to obtain the credentials that give access to the models with which the API works.

These models require images with the specifications of the iRAP projects. This means that they have been taken every 20 meters along the entire path to be analyzed. In addition, some of the models require images to be taken from the front and others from the side of the car. The models yield 1 result for each model for groups of 5 images or less. 

So far, 15 models compatible with the iRAP labeling specifications have been developed and are specified in the table below. 


| Model Name             | Description                                   | Type of Image | Classes |
|------------------------|---------------------------------------------- | ------------- | ------- |
| delineation            | Adequacy of road lines                        | Frontal       | 2       |
| street lighting        | Presence of street lighting                   | Frontal       | 2       |
| carriageway            | Carriageway label for section                 | Frontal       | 2       |
| service road           | Presence of a service road                    | Frontal       | 2       |
| road condition         | Condition of the road surface                 | Frontal       | 3       |
| skid resistance        | Skidding resistance                           | Frontal       | 3       |
| upgrade cost           | Influence surroundings on cost of major works | Frontal       | 3       |
| speed management       | Presence of features to reduce operating speed| Frontal       | 3       |
| bicycle facility       | Presence of facilities for bicyclists         | Frontal       | 2       |
| quality of curve       | Influence surroundings on cost of major works | Frontal       | 2       |
| vehicle parking        | Influence surroundings on cost of major works | Frontal       | 2       |
| property access points | Influence surroundings on cost of major works | Frontal       | 2       |
| area_type              | Influence surroundings on cost of major works | Lateral       | 2       |
| land use               | Influence surroundings on cost of major works | Lateral       | 4       |
| number of lanes        | Influence surroundings on cost of major works | Frontal       | 5       |

Some of the models can identify all the classes or categories, others can help you sort through the available options.

## Main Features

Some of the features now available are as follows:

- Scoring using the models already developed
- Grouping by groups of 5 images from an image list
- Download models directly into the root of the package

## Instalation

To install you can use the following commands

```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple viasegura_test==0.0.1.25

```

Then to download the models use the following commands

```
from viasegura_test import download_models

download_models(aws_access_key = <aws_access_key>, aws_secret_key = <aws_secret_key> )
```

To obtain the corresponding credentials for downloading the models, please contact the Inter-American Development Bank.

You can also clone the repository but remember that the package is configured to download the models and place them in the root of the environment. You can change the locations manually as follows

```
from viasegura_test import download_models

download_models(aws_access_key = <aws_access_key>, aws_secret_key = <aws_secret_key> , system_path = <new_working_path>)
```

Remember to put that path every time you instantiate a model so that you can find the artifacts you need to run them.