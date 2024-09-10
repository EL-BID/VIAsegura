[![Downloads](https://pepy.tech/badge/viasegura)](https://pepy.tech/project/viasegura)
<h1 align="center"> VIAsegura</h1>

## Content Table:
---

- [Project Description](#project-description)
- [Main Features](#main-features)
- [Installation](#installation)
- [Using the Models](#using-the-models)
- [User's guides](#user-guide)
- [Autores](#autores)
- [License](#license)



## Project Description
---

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
| quality of curve       | How adecuate is the curve                     | Frontal       | 2       |
| vehicle parking        | Presence of parking on the road               | Frontal       | 2       |
| property access points | Detects access to properties                  | Frontal       | 2       |
| area_type              | Detects if there is an urban or rural area    | Lateral       | 2       |
| land use               | Describes the use of the land surrounding the road | Lateral       | 4       |
| number of lanes        | The numebr of lanes detected                  | Frontal       | 5       |

Some of the models can identify all the classes or categories, others can help you sort through the available options.

## Main Features
---

Some of the features now available are as follows:

- Scoring using the models already developed
- Grouping by groups of 5 images from an image list
- Download models directly into the root of the package

## Instalation
---

To install you can use the following commands

```
pip install viasegura

```

Then to download the models use the following commands

```
from viasegura import download_models

download_models(url = <signed_url>)
```

Or alternativily

```
from viasegura import download_models

download_models(aws_access_key = <aws_access_key>, signature = <signature>, expires = <expiration_time>)
```


To obtain the corresponding credentials for downloading the models, please contact the Inter-American Development Bank at infradigital@iadb.org

You can also clone the repository but remember that the package is configured to download the models and place them in the root of the environment. You can change the locations manually as follows

```
from viasegura import download_models

download_models(url = <signed_url>, system_path = <new_working_path>)
```

Or alternativily

```
from viasegura import download_models

download_models(aws_access_key = <aws_access_key>, signature = <signature>, expires = <expiration_time>, system_path = <new_working_path>)
```


Remember to put that path every time you instantiate a model so that you can find the artifacts you need to run them.

## Using the Models
---

In order to make the instance of a model you can use the following commands

```
from viasegura import ModelLabeler

labeler = ModelLabeler(<type>) 
```

You can use either "frontal" or "lateral" tag in order to use the group of models desired (see table above)


For example, we can see both instance types:


```
from viasegura import ModelLabeler

frontal_labeler = ModelLabeler('frontal') 
```

or 

```
from viasegura import ModelLabeler

lateral_labeler = ModelLabeler('lateral') 
```

Also you can especify which models to load using the parameter model filter and the name of the models to use, (see the table above):

```
from viasegura import ModelLabeler

frontal_labeler = ModelLabeler('frontal', model_filter = ['delineation', 'street_lighting', 'carriageway']) 
```

In addition, you can make it work using the GPU specifying the device where the models are going to run, for example

```
from viasegura import ModelLabeler

frontal_labeler = ModelLabeler('frontal', device='/device:GPU:0') 
```
## User Guide

You can see and entire example of use on [this link](https://colab.research.google.com/drive/1WltAaCCOnREcuLzI3PX_IKsnHNuRttEz?usp=sharing).

Also make sure to see the [manual](https://github.com/EL-BID/VIAsegura/tree/main/viasegura/manuals) to understand the scope of the project and how to make a project from scratch using the viasegura models.

You can modify the devices used according to the TensorFlow documentation regarding GPU usage (see https://www.tensorflow.org/guide/gpu)

## Autores
---

This package has been developed by:

<a href="https://github.com/J0s3M4rqu3z" target="blank">Jose Maria Marquez Blanco</a>
<br/>
<a href="https://www.linkedin.com/in/joancerretani/" target="blank">Joan Alberto Cerretani</a>

## License

The distribution of this software is according with the following [license](https://github.com/EL-BID/VIAsegura/blob/main/LICENSE.md)
