<div align="center">
    <h1>VIAsegura</h1>
    <h3>Automatic labeling of road safety attributes</h3>

![analytics image (flat)](https://raw.githubusercontent.com/vitr/google-analytics-beacon/master/static/badge-flat.gif)
[![Downloads](https://pepy.tech/badge/viasegura)](https://pepy.tech/project/viasegura)
</div>

## Content Table:
---
- [Project Description](#project-description)
- [Main Features](#main-features)
- [Quick Start](#quick-start)
    - [Installation](#installation)
    - [Using the Models](#using-the-models)
- [User's guides](#users-guide)
- [Authors](#authors)
- [License](#license)

## Project Description
---

VIAsegura is a library that helps to use artificial intelligence models developed by the Inter-American Development Bank to
automatically tag items on the streets. The tags it places are some of those needed to implement the iRAP road safety
methodology.

These models require images with the specifications of the iRAP projects. This means that they have been taken every 20
meters along the entire path to be analyzed. In addition, some of the models require images to be taken from the front
and others from the side of the car. The models yield 1 result for each model for groups of 5 images or fewer.

So far, 15 models compatible with the iRAP labeling specifications have been developed and are specified in the table
below.

| Model Name             | Description                                        | Type of Image | Classes |
|------------------------|----------------------------------------------------|---------------|---------|
| delineation            | Adequacy of road lines                             | Frontal       | 2       |
| street_lighting        | Presence of street lighting                        | Frontal       | 2       |
| carriageway            | Carriageway label for section                      | Frontal       | 2       |
| service_road           | Presence of a service road                         | Frontal       | 2       |
| road_condition         | Condition of the road surface                      | Frontal       | 3       |
| skid_resistance        | Skidding resistance                                | Frontal       | 3       |
| upgrade_cost           | Influence surroundings on cost of major works      | Frontal       | 3       |
| speed_management       | Presence of features to reduce operating speed     | Frontal       | 3       |
| bicycle_facility       | Presence of facilities for bicyclists              | Frontal       | 2       |
| quality_of_curve       | How adequate is the curve                          | Frontal       | 2       |
| vehicle_parking        | Presence of parking on the road                    | Frontal       | 2       |
| property_access_points | Detects access to properties                       | Frontal       | 2       |
| area_type              | Detects if there is an urban or rural area         | Lateral       | 2       |
| land_use               | Describes the use of the land surrounding the road | Lateral       | 4       |
| number_of_lanes        | The number of lanes detected                       | Frontal       | 5       |

Some of the models can identify all the classes or categories, others can help you sort through the available options.

## Main Features
---

Some of the features now available are as follows:

- Scoring using the models already developed
- Grouping by groups of 5 images from an image list
- Download models directly into the root of the package

## Quick Start
---

### Installation

To install you can use the following commands

```bash
pip install viasegura
```

Then to download the models from [the repository](https://github.com/EL-BID/VIAsegura/raw/refs/heads/release/v2.0.0/models/models.tar.gz?download=) to ...

Remember to put that path every time you instantiate a model so that you can find the artifacts you need to run them.

### Using the Models

In order to make the instance of a model you can use the following commands

```python
from viasegura import ModelLabeler

labeler = ModelLabeler(<type>) 
```

You can use either "frontal" or "lateral" tag in order to use the group of models desired (see table above)

For example, we can see both instance types:

```python
from viasegura import ModelLabeler

frontal_labeler = ModelLabeler('frontal')  # or 'lateral' 
```

Also, you can specify which models to load using the parameter model filter and the name of the models to use, (see the
table above):

```python
from viasegura import ModelLabeler

frontal_labeler = ModelLabeler('frontal', model_filter=['delineation', 'street_lighting', 'carriageway']) 
```

In addition, you can make it work using the GPU specifying the device where the models are going to run, for example

```python
from viasegura import ModelLabeler

frontal_labeler = ModelLabeler('frontal', device='/device:GPU:0') 
```

## Users Guide

You can see and entire example of use on this 
[notebook](https://github.com/EL-BID/VIAsegura/blob/release/v2.0.0/notebooks/execution_example.ipynb) on 
'notebooks' folder.

Also make sure to see the [manual](https://github.com/EL-BID/VIAsegura/tree/main/viasegura/manuals) to understand the
scope of the project and how to make a project from scratch using the viasegura models.

You can modify the devices used according to the TensorFlow documentation regarding GPU usage (
see https://www.tensorflow.org/guide/gpu)

## Authors

This package has been developed by:

<a href="https://github.com/J0s3M4rqu3z" target="blank">Jose Maria Marquez Blanco</a>
<br/>
<a href="https://www.linkedin.com/in/joancerretani/" target="blank">Joan Alberto Cerretani</a>
<br/>
<a href="https://www.linkedin.com/in/ingvictordurand/" target="blank">Victor Durand</a>

## License

The distribution of this software is according to the
following [license](https://github.com/EL-BID/VIAsegura/blob/main/LICENSE.md)
