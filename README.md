<div align="center">
    <h1>VIAsegura</h1>
    <h3>Automatic labeling of road safety attributes</h3>

[Users manual](https://github.com/EL-BID/VIAsegura/blob/release/v2.0.0/docs/manual_viasegura.pdf)

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

### Environment Setup

It is advisable to install the library in a Python virtual environment. If you use `conda`, you can do so with the following command:

```bash
conda create -n viasegura python=3.10
conda activate viasegura
```

> Supported versions of Python are 3.9, 3.10, 3.11 and 3.12.
> For more information on how to install conda, you can visit the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

### Installation

To install the package you can use the following commands in the terminal 

```bash
pip install viasegura
```

To download the models use [this link from the repostitory](https://github.com/EL-BID/VIAsegura/raw/refs/heads/release/v2.0.0/models/models.tar.gz?download=).
Put the downloaded file on a `models` folder and decompress it with the following command:

```bash
tar -xzvf models.tar.gz
```
The path where the models are must be `models/models_artifacts` 
Remember to put that path every time you instantiate a model so that you can find the artifacts you need to run them.

### Using the Models

In order to make the instance of a model you can use the following commands

```python
from viasegura import ModelLabeler

frontal_labeler = ModelLabeler('frontal')  # or 'lateral' 
```

You can use either "frontal" or "lateral" tag in order to use the group of models desired (see table above)

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
> You can modify the devices used according to the [TensorFlow documentation](https://www.tensorflow.org/guide/gpu) 
> regarding GPU usage.

For a full example of use on this package, you can see this 
[notebook](https://github.com/EL-BID/VIAsegura/blob/release/v2.0.0/notebooks/execution_example.ipynb).
In the `notebooks` folder you can also find example images to test the execution of the models.


## Users Guide

I invite you to visit the [manual](https://github.com/EL-BID/VIAsegura/blob/release/v2.0.0/docs/manual_viasegura.pdf) to understand the
scope of the project and how to make a project from scratch using the viasegura models.


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


## Acknowledgments / Reconocimientos

**Copyright © [2025]. Inter-American Development Bank ("IDB"). Authorized Use.**  
The procedures and results obtained based on the execution of this software are those programmed by the developers and do not necessarily reflect the views of the IDB, its Board of Executive Directors or the countries it represents.

**Copyright © [2025]. Banco Interamericano de Desarrollo ("BID"). Uso Autorizado.**  
Los procedimientos y resultados obtenidos con la ejecución de este software son los programados por los desarrolladores y no reflejan necesariamente las opiniones del BID, su Directorio Ejecutivo ni los países que representa.

### Support and Usage Documentation / Documentación de Soporte y Uso

**Copyright © [2025]. Inter-American Development Bank ("IDB").** The Support and Usage Documentation is licensed under the Creative Commons License CC-BY 4.0 license. The opinions expressed in the Support and Usage Documentation are those of its authors and do not necessarily reflect the opinions of the IDB, its Board of Executive Directors, or the countries it represents.

**Copyright © [2025]. Banco Interamericano de Desarrollo (BID).** La Documentación de Soporte y Uso está licenciada bajo la licencia Creative Commons CC-BY 4.0. Las opiniones expresadas en la Documentación de Soporte y Uso son las de sus autores y no reflejan necesariamente las opiniones del BID, su Directorio Ejecutivo ni los países que representa.

### AI-Powered Services Disclaimer / Exención de responsabilidad por Servicios Impulsados por IA

The Software may include features which use, are powered by, or are an artificial intelligence system (“AI-Powered Services”), and as a result, the services provided via the Software may not be completely error-free or up to date. Additionally, the User acknowledges that due to the incorporation of AI-Powered Services in the Software, the Software may not dynamically (in “real time”) retrieve information and that, consequently, the output provided to the User may not account for events, updates, or other facts that have occurred or become available after the Software was trained. Accordingly, the User acknowledges that the use of the Software, and that any actions taken or reliance on such products, are at the User’s own risk, and the User acknowledges that the User must independently verify any information provided by the Software.

El Software puede incluir funciones que utilizan, están impulsadas por o son un sistema de inteligencia artificial (“Servicios Impulsados por IA”) y, como resultado, los servicios proporcionados a través del Software pueden no estar completamente libres de errores ni actualizados. Además, el Usuario reconoce que, debido a la incorporación de Servicios Impulsados por IA en el Software, este puede no recuperar información dinámicamente (en “tiempo real”) y que, en consecuencia, la información proporcionada al Usuario puede no reflejar eventos, actualizaciones u otros hechos que hayan ocurrido o estén disponibles después del entrenamiento del Software. En consecuencia, el Usuario reconoce que el uso del Software, y que cualquier acción realizada o la confianza depositada en dichos productos, se realiza bajo su propio riesgo, y reconoce que debe verificar de forma independiente cualquier información proporcionada por el Software.
