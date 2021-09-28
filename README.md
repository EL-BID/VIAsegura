# VIAsegura-test

## Project Description

VIAsegura is an API that helps to use artificial intelligence models developed by the Inter-American Development Bank to automatically tag items on the streets. The tags it places are some of those needed to implement the iRAP road safety methodology. 

To use it you must contact the Inter-American Development Bank to obtain the credentials that give access to the models with which the API works.

These models require images with the specifications of the iRAP projects. This means that they have been taken every 20 meters along the entire path to be analyzed. In addition, some of the models require images to be taken from the front and others from the side of the car. The models yield 1 result for each model for groups of 5 images or less. 

So far, 16 models compatible with the iRAP labeling specifications have been developed and are specified in the table below. 


| Model Name      | Description                                   | Type of Image | Classes |
|-----------------|---------------------------------------------- | ------------- | ------- |
| delineation     | Adequacy of road lines                        | Frontal       | 2       |
| street lighting | Presence of street lighting                   | Frontal       | 2       |
| carriageway     | Carriageway label for section                 | Frontal       | 2       |
| service road    | Presence of a service road                    | Frontal       | 2       |
| road condition  | Condition of the road surface                 | Frontal       | 3       |
| skid resistance | Skidding resistance                           | Frontal       | 3       |
| upgrade cost    | Influence surroundings on cost of major works | Frontal       | 3       |