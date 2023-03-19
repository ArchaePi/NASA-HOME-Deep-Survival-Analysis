# <img src="https://drive.google.com/uc?export=view&id=1XoL0DKY9N63l-rIop8PnYnHXGRz_AOfx" width="100" height="100"> NASA HOME Deep Survival Analysis
<br>


***
<br>


## Table of Contents
1. [Overview](#overview)
   * [What is NASA HOME?](#nasa-home)
   * [What are Deep Space Habitats (DSH)?](#dsh)
   * [What is Deep Survival Analysis for Predictive Maintenance?](#dsa)
   * [CMU Specific NASA HOME Research Goals](#cmu)
2. [Data](#data)
   * [NASA's Li-ion Battery Aging Datasets](#li-ion)
3. [Methodology](#method)
   * [Preprocessing](#pre-proc)
   * [Training](#train)
4. [Results](#results)
   * [Performance Metrics](#metrics)
   * [Graphs](#graphs)
5. [Discussion and Conclusions](#disc-cons)
   * [Limitations](#limits)
   * [Future Work](#future)
6. [Resources and References](#res-ref)
7. [Collaborators](#collab)


<br>

***
<br>


<a name = "overview"></a>
## Overview
<a name = "nasa-home"></a>
### What is NASA HOME?
<b>Habitats Optimized for Missions of Exploration (HOME)</b> is a multi-university, multi-disciplinary Space Technology Research Institute (STRI) and university-led research project proposed by NASA. HOME was proposed for the purposes of developing optimal designs of Deep Space Habitats (DSH). A DSH requires the synthesis of reliability engineering, risk analysis, and a myriad of emergent technologies which enable the development of robust, autonomous, and self-dependent habitats. By optimizing the designs of DSH, we propagate ourselves towards a future where reliable, and safe habitats for long-term deep space exploration, can be a space ;) astronauts call home.

[NASA's Vision Statement](https://www.nasa.gov/sites/default/files/atoms/files/home_quad_chart.pdf)
<br>

<a name = "dsh"></a>
### What are Deep Space Habitats (DSH)?
<b>Deep Space Habitats (DSH)</b> are a conceptualized habitat proposed by NASA. These habitats are intended to maintain an extended residence in deep space and provide a safe, self-dependent environment for human explorers during prolonged deep space missions. The term 'deep space' includes spaces which extend beyond low-Earth orbit (LEO) and cislunar space, onward to Mars and other unexplored regions of our solar system. DSHs can be simplified as consisting of the two following parts: deep space habitation systems and habitation modules.
<br><br>
<img src="https://drive.google.com/uc?export=view&id=16flpdqapmh1jcujg2csmyAEqBFnsk9hw" width="320" height="180">
<br>
A conceptual depiction of a deep space habitat's interior configuration complete with plant growth chambers, research stations, and a robtic helper. <b>Image Source: NASA</b>

#### Habitation Systems
Deep space habitation systems are systems which provide life support, environmental monitoring, crew health maintenance, fire safety, radiation protection, as well as, many other elements intended to prolong the life of both the DSH and its inhabitants during its residence in deep space. 
#### Habitation Modules
Habitation Modules are the structures which house both the aforementioned habitation systems, and future human explorers. These modules require optimal and careful construction considering a module’s obligation to protect astronauts from extreme deep space conditions.
<br>

<a name = "dsa"></a>
### What is Deep Survival Analysis for Predictive Maintenance?
#### Survival Analysis
Survival Analysis (also known as time-to-event analysis) is a set of statistical methods implemented to estimate the time until an event has occurred.

#### Predictive Maintenance
Predictive Maintenance is an application of survival analysis, where the event of interest is the breakdown or failure of a machine or mechanical part.
<br>

<a name = "cmu"></a>
### CMU Specific NASA HOME Research Goals
#### University
* "<b>Root cause analysis of faults in the Environmental Control and Life Support System (ECLSS) and the Electrical Power System (EPS).</b> We will be investigating the integration of a digital twin model with root cause analysis methods to enable context-based diagnosis of faults." - [CMU MOSAIC Projects](https://faculty.ce.cmu.edu/mosaic/projects/habitats-optimized-for-missions-of-exploration-nasa-home/)
* "<b>Uncertainty quantification in digital twins.</b> For safety-critical systems, predictions of system health that account for uncertainties in a robust and efficient manner are deemed essential by NASA to safeguard against failures. Our principal objective is to develop approaches for effectively quantifying and representing uncertainties in digital twins." - [CMU MOSAIC Projects](https://faculty.ce.cmu.edu/mosaic/projects/habitats-optimized-for-missions-of-exploration-nasa-home/)
#### Deep Survival Analysis Researchers:
An optimal, and self-dependent DSH requires early prediction of system shutdowns and failures. Therefore, to achieve the research objectives of NASA HOME, we will be training and developing deep survival models in order to estimate the remaining useful life (RUL) of deep space habitation system components. 
<br><br>
[The auton-survival package](https://github.com/autonlab/auton-survival)
<br><br>


<a name = "data"></a>
## Data
<a name = "li-ion"></a>
### NASA's Li-ion Battery Aging Datasets
"This data set has been collected from a custom built battery prognostics testbed at the NASA Ames Prognostics Center of Excellence (PCoE). Li-ion batteries were run through 3 different operational profiles (charge, discharge and Electrochemical Impedance Spectroscopy) at different temperatures. Discharges were carried out at different current load levels until the battery voltage fell to preset voltage thresholds. Some of these thresholds were lower than that recommended by the OEM (2.7 V) in order to induce deep discharge aging effects. Repeated charge and discharge cycles result in accelerated aging of the batteries. The experiments were stopped when the batteries reached the end-of-life (EOL) criteria of 30% fade in rated capacity (from 2 Ah to 1.4 Ah)." - [NASA's Open Data Portal](https://data.nasa.gov/dataset/Li-ion-Battery-Aging-Datasets/uj5r-zjdb)
<br><br>


<a name = "method"></a>
## Methodology
<a name = "pre-proc"></a>
### Preprocessing
<br>

<a name = "train"></a>
### Training
<br><br>


<a name = "results"></a>
## Results
<a name = "metrics"></a>
### Performance Metrics
<br>

<a name = "graphs"></a>
### Graphs
<br><br>


<a name = "disc-cons"></a>
## Discussion and Conclusions
<a name = "limits"></a>
### Limitations
<br>

<a name = "future"></a>
### Future Work
<br><br>


<a name = "res-ref"></a>
## Resources and References
* [Deep Space Habitats Overview](https://www.nasa.gov/deep-space-habitation/overview/)
* [Habitats Optimized for Missions of Exploration (HOME)](https://www.nasa.gov/directorates/spacetech/strg/stri/stri_2018/Habitats_Optimized_for_Missions_of_Exploration_HOME/)
* [HOME Website](https://homestri.ucdavis.edu/)
* [NASA's Quad Chart](https://www.nasa.gov/sites/default/files/atoms/files/home_quad_chart.pdf)
<br><br>



<a name = "collab"></a>
## Collaborators
[Shakirah D. Cooper](https://github.com/ArchaePi) - MS Computational Biomedical Engineering @ Carnegie Mellon University
<br><br>


***
