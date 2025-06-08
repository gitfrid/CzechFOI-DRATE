# CzechFOI-DRATE

CzechFOI-DRATE: Exploring ways to minimize bias when dividing real-world data into two groups (vaccinated vx /unvaccinated uvx)
<br>

**Hypothesis:
It is impossible to perfectly and fairly compare vaccinated (VX) and unvaccinated (UVX) groups — either by measurement or mathematically — when vaccination is time-dependent and not random. This remains true if both groups have the same homogen individual death rates.**

<br>
_________________________________________

### When comparing different methods, Cox PH seemed to calculate the best approximation

<br>Phyton script [W) coxph real deaths real vax dates by age](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age.py)
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age.html)
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age.png width="1280" height="auto">
<br>
<br>
To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age/W%29%20coxph%20no%20bias%20sim%20deaths%20sim%20vax%20dates%20by%20age.png width="1280" height="auto">
<br>

_________________________________________

### DoWhy causal impact estimation

<br>Phyton script [ZA) dowhy doses vs sim_total_death individual.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/ZA%29%20dowhy%20doses%20vs%20sim_total_death%20individual.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZA%29%20dowhy%20doses%20vs%20total_death%20individual/ZA%29%20doses_vs_total_deaths_dowhy_individual.html)
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZA)%20dowhy%20doses%20vs%20total_death%20individual/ZA)%20doses_vs_total_deaths_dowhy_individual.png width="1280" height="auto">
<br>
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZA)%20dowhy%20doses%20vs%20total_death%20individual/ZA)%20sim%20HR%20no%20bias%20doses_vs_total_deaths_dowhy_individual.png width="1280" height="auto">
<br>

_________________________________________

### Following a lot of other analyses
________________________________________

### ZF) vx uvx norm

<br>Phyton script [ZF) vx uvx norm.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/ZF%29%20vx%20uvx%20norm.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZF%29%20vx%20uvx%20norm/ZF%29%20vx%20uvx%20norm.html)
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZF)%20vx%20uvx%20norm/ZF)%20vx%20uvx%20norm.png width="1280" height="auto">
<br>
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZF)%20vx%20uvx%20norm/ZF)%20vx%20uvx%20norm%20sim%20no%20bias.png width="1280" height="auto">
<br>

_________________________________________

### E) Death risk by age over time

<br>Phyton script [E) death risk by age.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/E%29%20death%20risk%20by%20age.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E%29%20death%20risk%20by%20age/E%29%20vx_uvx_death_risk_by_age.html)
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E%29%20death%20risk%20by%20age/E%29%20vx_uvx_death_risk_by_age.png width="1280" height="auto">
<br>
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E%29%20death%20risk%20by%20age/E%29%20no%20bias%20vx_uvx_death_risk_by_age.png width="1280" height="auto">
<br>

_________________________________________
### E) Death risk by age over time

<br>Phyton script [E) death risk by age.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/E%29%20death%20risk%20by%20age.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E%29%20death%20risk%20by%20age/E%29%20vx_uvx_death_risk_by_age.html)
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E%29%20death%20risk%20by%20age/E%29%20vx_uvx_death_risk_by_age.png width="1280" height="auto">
<br>
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E%29%20death%20risk%20by%20age/E%29%20no%20bias%20vx_uvx_death_risk_by_age.png width="1280" height="auto">
<br>
_________________________________________

### Software Requirements:

These scripts don't require SQLite queries to aggregate the 11 million individual data rows.
Instead, the aggregation is handled directly by Python scripts, which can generate aggregated CSV files very quickly.

- [Python 3.12.5](https://www.python.org/downloads/) to run the scripts.
- [Visual Studio Code 1.92.2](https://code.visualstudio.com/download) to edit and run scripts.


### Disclaimer:
**The results have not been checked for errors. Neither methodological nor technical checks or data cleansing have been performed.**


