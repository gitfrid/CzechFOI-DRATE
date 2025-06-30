# CzechFOI-DRATE

CzechFOI-DRATE: Exploring ways to minimize bias when dividing real-world data into two groups (vaccinated vx /unvaccinated uvx)


_________________________________________

### Hypothesis 1 - [ CzechFOI-StackSim repository](https://github.com/gitfrid/CzechFOI-StackSim): <br><br>It is impossible to perfectly and fairly compare vaccinated (VX) and unvaccinated (UVX) groups — either by measurement or mathematically — when vaccination is time-dependent and not random. <br>This remains true if both groups have the same homogen individual constant death rates.

_________________________________________

### Hypothesis 2 - [see CzechFOI-SIM repository](https://github.com/gitfrid/CzechFOI-SIM):<br><br>There is currently no reliable statistical method to determine the rate of death-related Adverse Events Following Immunisation (dAEFIs) at a frequency of approximately one additional death per 10,000 doses when the baseline mortality is unknown in real-world settings.
<br>**To the best of my knowledge,  this (vital) problem is still waiting for the head that can solve it, as a benefactor of mankind?
This also applies vice versa (one death per 10,000 doses removed/saved)**

_________________________________________

### Project GOAL<br><br> The aim is to find a method that compensates for biases introduced by the non-random assignment of individuals to vaccinated (vx) and unvaccinated (uvx) groups based on the timing of vaccination. This type of bias is unavoidable in real-world datasets, but it must be corrected in order to enable a fair comparison between the two groups.

Simulated Test Dataset with Minimal Bias

A synthetic dataset was generated in which individuals within each age group share a constant and homogeneous risk of death, estimated from real-world age-specific death rates. Death dates were simulated independently of vaccination status.
Real-world vaccination schedules (dose sets) were then reassigned randomly to individuals within the same age group, ensuring that each individual's entire dose schedule occurred on or before their simulated death date. No actual death dates were removed or altered—only the assignment of dose dates was adjusted to maintain this temporal consistency. This approach minimizes bias while preserving realistic dose timing patterns.
<br>
<br>sim_MINBIAS_Vesely_106_SIMULATED.csv created by [NK) generate csv simulate deaths minimal bias.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/NK%29%20generate%20csv%20simulate%20deaths%20minimal%20bias.py)
<br>30.06.2025 changed constraint death day >= last dose day to death day > last dose day  as cox can't handle zero intervalls where start = stop 
________________________________________________

### G) G-estimate and Cox time variing methode to compensate for bias - Hypothsis 1
<br>Test using simulated dataset based on real world paramter created by "NK) generate csv simulate deaths minimal bias.py". 
<br>Tried to evaluate whether the G-estimation (psi) method could correct for bias, but struggled with error messages.
<br>By using the parameter of "Cox time variing methode" below it should probably work.
<br>
<br>Phyton script [G) generate interval data per person.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/G%29%20generate%20interval%20data%20per%20person.py)
<br>Phyton script [G) G-estimation on interval data per person.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/G%29%20G-estimation%20on%20interval%20data%20per%20person.py)
<br>Phyton script [GA) G-estimation on interval data per person all age.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/GA%29%20G-estimation%20on%20interval%20data%20per%20person%20all%20age.py)
<br>Phyton script [G) cox on interval data per person.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/G%29%20cox%20on%20interval%20data%20per%20person.py)
<br>

_________________________________________
### AB) Testscript for Cox time variing methode bias correction AGE 70 - Hypothsis 1
<br>
Impact of Dose Assignment Strategy on bias correction and Estimated Mortality Risk

Objective
To assess how different vaccine dose assignment strategies affect estimated hazard ratios (HRs) for mortality.

Methods
Time-varying Cox regression was used to compare mortality risk between vaccinated and unvaccinated individuals under three scenarios:
All three simulated with random homogen and constant death probability for the whole population.

    Case 1: Simulated doses with the same distribution as the Real-world Czech FOI data (death must follow dose) - 
            The csv dataset was created by "NK) generate csv simulate deaths minimal bias.py" based on the real world data
            (see Project GOAL).
            
    Case 2: Simulated doses with flat random assignment (death must follow dose).

    Case 3: Simulated doses with a bell-curve distribution(death must follow dose).

Results
HRs varied by dose assignment logic, highlighting the impact of immortal time bias.

    Case 1: Simulation based on Real-world dataset parameter produced HR ≈ 0.51 (≈49% mortality reduction), but the cause of this effect requires further investigation.
    
    Case 2: Flat random assignment yielded HRs ≈ 1.0 (no effect).

    Case 3: Bell-curve distribution logic led to inflated HRs (>1.0).    

<br>Conclusion
Dose classification strategies strongly influence observed vaccine effectiveness. Careful control of timing and classification is essential to avoid bias in survival analyses.
**Result of CASE 1) requires further investigation!!**
<br>
<br>Phyton script [AB) Cox fair compare vx uvx.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/AB%29%20Cox%20fair%20compare%20vx%20uvx.py)
<br>
<br>
 _________________________________________

<br>

### X) Normalized UVx/Vx comparison and stacked events plot - Hypothesis 1
<br>Plots generated by Phyton script [X) event_stacking.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/X%29%20event_stacking.py) . Here you can [Download the interactive htmls](https://github.com/gitfrid/CzechFOI-DRATE/tree/main/Plot%20Results/X%29%20event_stacking)
<br>
<br>
<br>**Plot of simulated dataset below assuming a homogeneous, uniform, and time-invariant mortality rate across age groups (at about real world level). Afterward Individuals were randomly assigned to vaccinated or unvaccinated cohorts, with real-world dosing schedules applied. Enforcing that death could only occur post-vaccination (real world scenario) inherently introducing immortal time bias, as illustrated below.
<br>
<br>As a reminder, every individual in the homogeneous population below has the same constant mortality risk. If group assignment is non-random (as occurs in the real world), this introduces bias, making the normalized mortality rate of the UVX group looks much worse.**
<br>
<br>Test dataset sim_MINBIAS_Vesely_106_SIMULATED.csv
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/X)%20event_stacking/X%29%20sim%20MINBIAS%20Population_and_Deaths_Trends_with_All_Doses.png width="1280" height="auto">
<br>The stacked events for the MINBIAS simulation AG 70 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/X)%20event_stacking/X)%20sim%20MINBIAS%20DoseAligned_Stacked_Normalized_Deaths.png width="1280" height="auto">
<br>
<br>
<br>**Normalized Vx/Uvx death rate plot of Czech real world data AG 70**
<br>Czech real world dataset Vesely_106_202403141131.csv [Download Freedom of Information Request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data)
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/X)%20event_stacking/X%29%20Population_and_Deaths_Trends_with_All_Doses.png width="1280" height="auto">
<br>The stacked events for czech reqal world data AG 70
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/X)%20event_stacking/X%29%20DoseAligned_Stacked_Normalized_Deaths.png width="1280" height="auto">
<br>
<br>
<br>**Plot of simulated dataset estimates age-specific death rates and simulates constant, uniformly random death dates across the observation window, preserving vaccination timings but ignoring cases where death precedes vaccination to prevent any bias - not the case in real world scenario.**
<br>
<br>Test dataset sim_NOBIAS_Vesely_106_202403141131.csv created by [NC) generate csv simulate deaths no bias.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/NC%29%20generate%20csv%20simulate%20deaths%20no%20bias.py)
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/X)%20event_stacking/X%29%20sim%20NOBIAS%20Population_and_Deaths_Trends_with_All_Doses.png width="1280" height="auto">
<br>
<br>The stacked events AG 70 for the NOBIAS simulation should theoretically run horizontally at about the same level — could be a bug in the code.
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/X)%20event_stacking/X)%20sim%20NOBIAS%20DoseAligned_Stacked_Normalized_Deaths.png width="1280" height="auto">
<br>

_________________________________________

### R) Baseline-Normalized Excess Mortality Analysis Aligned to Vaccination Doses methode for Hypothsis 1

<br>Phyton script [R) EM baseline_rate_norm.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/R%29%20EM%20baseline_rate_norm.py)
<br>
<br>Attempt to replicate the method from the original R code using Python.
<br>The R implementation from @henjin256 is documented here: https://sars2.net/czech2.html#Excess_mortality_by_weeks_after_vaccination
<br>
<br>**So far, I have not been able to reproduce the same results — likely due to methodical or logical errors in my Python code.
When I translated the R code line-by-line into Python, it produced massive tables that exceeded the capacity of RAM or Python to handle, so I had to adjust the method**
<br>
I tried to use a different input format compared to the R version.
The used input file "C:\CzechFOI-DRATE\intervals_per_agebin\real_interval_person_all_ages_Vesely_106_202403141131.csv" was generated from the original Czech Veselý dataset using the script: 
[G) generate interval data per person.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/G%29%20generate%20interval%20data%20per%20person.py) 
<br>

<br>**The result below is not plausible and does not match with @henjin256's r-code result** 
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/R)%20EM%20Baseline%20Rate%20Normalization/R)%20EM%20Baseline%20Rate%20Normalization.png width="1280" height="auto">
<br>

### AA) Time-since-first-dose person-time stratification

**Tried again to recode the methode used by r-code @henjin256's**
<br>https://sars2.net/czech2.html#Excess_mortality_by_weeks_after_vaccination
<br>https://sars2.net/czech.html#Bucket_analysis

<br>Creates buckets csv file for AGE 70 : [AA) generate bucket csv.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/AA%29%20generate%20bucket%20csv.py) (takes 1,5 hours only for AG 70)
 <br>Create the plot file : [AA) AG70 sim MINBIAS record_level_mort_vx uvx.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/AA%29%20AG70%20sim%20MINBIAS%20record_level_mort_vx%20uvx.py)
<br>
<br>**If the method correctly adjusts for bias, the vaccinated excess mortality curve should be flat, at or near 0%**
<br>Since that is not observed:
    <br> - The MINBIAS test data or the underlying assumptions are incorrect,
    <br> - The method was not reproduced correctly
    <br> - There are errors in my code
<br>If none of these, then the method probably does not properly adjust for the bias as intended in the project goals
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/AA)%20record%20level%20mort/AA%29%20AG70%20sim%20MINBIAS%20record_level_mort_vx%20uvx.png width="1280" height="auto">
<br>

_________________________________________

### CA) person days  landmark methode for Hypothsis 1

<br>Phyton script [CA) Landmark adjust resampling truncation bias.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/CA%29%20Landmark%20adjust%20resampling%20truncation%20bias.py)
<br>
<br>Person days real world dataset Vesely_106_202403141131.csv 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/CA%29%20Landmark%20adjust%20resampling%20truncation%20bias/CA%29%20real%20RR_AGEBIN_Landmark_Comparison.png width="1280" height="auto">
<br>
<br>Uses a simulated minbias dataset to test whether the method compensates for mortality bias (forced restriction death_day >= last_dose_day). It seems to only partially correct the bias. RR should theoretically run horizontally at about 1 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/CA%29%20Landmark%20adjust%20resampling%20truncation%20bias/CA%29%20minbias%20RR_AGEBIN_Landmark_Comparison.png width="1280" height="auto">
<br>
<br>Uses simulated nobias dataset to test methode (no constraint death_day >= last_dose_day). RR should theoretically run horizontally at about 1
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/CA%29%20Landmark%20adjust%20resampling%20truncation%20bias/CA%29%20nobias%20RR_AGEBIN_Landmark_Comparison.png width="1280" height="auto">
<br>
 _________________________________________

### Y) person days  methode for Hypothsis 1

<br>Phyton script [Y) vx uvx persondays immortal time adjusted.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/Y%29%20vx%20uvx%20persondays%20immortal%20time%20adjusted.py)
<br>
<br> Person days real world dataset Vesely_106_202403141131.csv 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y)%20vx%20uvx%20persondays%20baslinemortality/Y)%20vx%20uvx%20personday_exess_mortality_adjusted.png width="1280" height="auto">
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y%29%20vx%20uvx%20persondays%20baslinemortality/Y)%20vx%20uvx%20personday_exess_mortality_adjusted_exess.png width="1280" height="auto">
<br>
<br>test dataset sim_NOBIAS_Vesely_106_202403141131.csv
<br>
<br>**To test for bias, I run the same code on simulated data with a homogen uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, to avoid any selection bias**.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y%29%20vx%20uvx%20persondays%20baslinemortality/Y%20sim%20NOBIAS%20vx%20uvx%20personday_exess_mortality_adjusted.png width="1280" height="auto">
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y%29%20vx%20uvx%20persondays%20baslinemortality/Y%20sim%20NOBIAS%20vx%20uvx%20personday_exess_mortality_adjusted_exess.png width="1280" height="auto">
<br>
test dataset sim_MINBIAS_Vesely_106_SIMULATED.csv.
<br>
<br>**The test dataset assuming a homogeneous, uniform, and time-invariant mortality rate across age groups (at about real world level). Afterward Individuals were randomly assigned to vaccinated or unvaccinated cohorts, with real-world dosing schedules applied. Enforcing that death could only occur post-vaccination (real world) inherently introducing immortal time bias, as illustrated below.
<br> Should be the same for vx uvx - perhaps bug in code or method not correct applied, or personday method don't adjust for death day >= last dose day bias?**
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y%29%20vx%20uvx%20persondays%20baslinemortality/Y%20sim%20MINBIAS%20vx%20uvx%20personday_exess_mortality_adjusted.png width="1280" height="auto">
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y%29%20vx%20uvx%20persondays%20baslinemortality/Y%20sim%20MINBIAS%20vx%20uvx%20personday_exess_mortality_adjusted_exess.png width="1280" height="auto">
<br>


<br>Phyton script [Y) vx uvx persondays.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/Y%29%20vx%20uvx%20persondays.py)
<br>
<br> Person days real world dataset Vesely_106_202403141131.csv 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y)%20vx%20uvx%20persondays/Y%29%20vx%20uvx%20persondays%20mortality.png width="1280" height="auto">
<br>
<br>simulated test dataset sim_NOBIAS_Vesely_106_202403141131.csv
<br>
<br>**To test for bias, I run the same code on simulated data with a homogen uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, to avoid any selection bias**.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y)%20vx%20uvx%20persondays/Y%29%20sim%20NOBIAS%20vx%20uvx%20persondays%20mortality.png width="1280" height="auto">
<br>
simulated test dataset sim_MINBIAS_Vesely_106_SIMULATED.csv.
<br>
<br>**The test dataset assuming a homogeneous, uniform, and time-invariant mortality rate across age groups (at about real world level). Afterward Individuals were randomly assigned to vaccinated or unvaccinated cohorts, with real-world dosing schedules applied. Enforcing that death could only occur post-vaccination (real world) inherently introducing immortal time bias, as illustrated below.
<br> Should be the same for vx uvx - perhaps bug in code or method not correct applied, or personday method don't adjust for death day >= last dose day bias?**
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y)%20vx%20uvx%20persondays/Y%29%20sim%20MINBIAS%20vx%20uvx%20persondays%20mortality.png width="1280" height="auto">
<br>


<br>Phyton script [Y) vx uvx persondays baslinemort.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/Y%29%20vx%20uvx%20persondays%20baslinemort.py)
<br>
<br> Person days real world dataset Vesely_106_202403141131.csv 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y%29%20vx%20uvx%20persondays%20baslinemortality/Y%20vx%20uvx%20persondays%20baslinemortality.png width="1280" height="auto">
<br>
<br>
<br>**To test for bias, I run the same code on simulated data with a homogen uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, to avoid any selection bias**.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y%29%20vx%20uvx%20persondays%20baslinemortality/Y%20sim%20NOBIAS%20vx%20uvx%20persondays%20baslinemortality.png width="1280" height="auto">
<br>
<br>
<br>**The test dataset assuming a homogeneous, uniform, and time-invariant mortality rate across age groups (at about real world level). Afterward Individuals were randomly assigned to vaccinated or unvaccinated cohorts, with real-world dosing schedules applied. Enforcing that death could only occur post-vaccination (real world) inherently introducing immortal time bias, as illustrated below.
<br> Should be the same for vx uvx - perhaps bug in code or method not correct applied, or personday method don't adjust for death day >= last dose day bias?**
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/Y%29%20vx%20uvx%20persondays%20baslinemortality/Y%20sim%20MINIBIAS%20vx%20uvx%20persondays%20baslinemortality.png width="1280" height="auto">
<br>

_________________________________________
<br>

### W) When comparing different methods, Cox PH seemed to calculate the best approximation for Hypothsis 1

<br>Phyton script [W) coxph real deaths real vax dates by age](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age.py) Here you can [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age.html)
<br>
<br> Cox PH analysis using Czech-FOI real world dataset Vesely_106_202403141131.csv 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age.png width="1280" height="auto">
<br>
<br>test dataset sim_NOBIAS_Vesely_106_202403141131.csv
<br>
<br>**To test for bias, I run the same code on simulated data with a homogen uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, to avoid any selection bias**.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/W)%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age/W%29%20coxph%20no%20bias%20sim%20deaths%20sim%20vax%20dates%20by%20age.png width="1280" height="auto">
<br>
### If the code below is correct, this might explain why scientists endlessly debate the results of their comparison
simulated test dataset sim_MINBIAS_Vesely_106_SIMULATED.csv.
<br>
<br>**The test dataset assuming a homogeneous, uniform, and time-invariant mortality rate across age groups (at about real world level). Afterward Individuals were randomly assigned to vaccinated or unvaccinated cohorts, with real-world dosing schedules applied. Enforcing that death could only occur post-vaccination (real world) inherently introducing immortal time bias, as illustrated below.**
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/W%29%20coxph%20real%20deaths%20real%20vax%20dates%20by%20age/W%29%20sim%20MINBIAS%20coxph%20deaths%20sim%20vax%20dates%20by%20age.png width="1280" height="auto">
<br>

_________________________________________

### ZA) DoWhy causal impact estimation

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

### B)  DeathRatesBy Age from aggregated

<br>
<br>This produces the same results as "ZF) vx uvx norm" above, but uses aggregated CSV files for the calculations.

The aggregated CSV files were generated using the Python script: [A) generate aggregated csv files from CzechFOI.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/A%29%20generate%20aggregated%20csv%20files%20from%20CzechFOI.py) 
<br>
<br>The data is then plotted with the Python script: [B)  DeathRatesByAge from aggregated.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/ZF%29%20vx%20uvx%20norm.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZF%29%20vx%20uvx%20norm/ZF%29%20vx%20uvx%20norm.html)
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/B%29%20DeathRates%20byage%20from%20aggregated%20csvfiles/B%29%20DeathRates%20PerAgeGroup.png width="1280" height="auto">
<br>
<br>

_________________________________________

### E) Death risk by age over time

<br>Phyton script [E) death risk by age.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/E%29%20death%20risk%20by%20age.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E%29%20death%20risk%20by%20age/E%29%20vx_uvx_death_risk_by_age.html)
<br>
<br>Czech-FOI real world dataset Vesely_106_202403141131.csv
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E%29%20death%20risk%20by%20age/E%29%20vx_uvx_death_risk_by_age.png width="1280" height="auto">
<br>
<br>Simulated test dataset sim_NOBIAS_Vesely_106_202403141131.csv.
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E%29%20death%20risk%20by%20age/E%29%20sim%20no%20bias%20vx_uvx_death_risk_by_age.png width="1280" height="auto">
<br>
<br>Simulated test dataset sim_MINBIAS_Vesely_106_SIMULATED.csv.
<br>The dataset assuming a homogeneous, uniform, and time-invariant mortality rate across age groups (at about real world level). Afterward Individuals were randomly assigned to vaccinated or unvaccinated cohorts, with real-world dosing schedules applied. Enforcing that death could only occur post-vaccination (real world) inherently introducing immortal time bias, as illustrated below.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/E)%20death%20risk%20by%20age/E%29%20sim%20MINBIAS%20vx_uvx_death_risk_by_age.png width="1280" height="auto">
<br>
_________________________________________

### F) daily crude HR vx/uvx

<br>Phyton script [F) rolling daily crude HR by age.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/F%29%20rolling%20daily%20crude%20HR%20by%20age.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/F%29%20rolling%20daily%20crude%20HR%20by%20age/F%29%20rolling%20daily%20crude%20HR%20by%20age.html)
<br>
<br><br>Czech real world dataset Vesely_106_202403141131.csv [Download Freedom of Information Request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data)
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/F%29%20rolling%20daily%20crude%20HR%20by%20age/F%29%20rolling%20daily%20crude%20HR%20by%20age.png width="1280" height="auto">
<br>
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/F%29%20rolling%20daily%20crude%20HR%20by%20age/F%29%20sim%20no%20bias%20rolling%20daily%20crude%20HR%20by%20age.png width="1280" height="auto">
<br>
<br>Simulated test dataset sim_MINBIAS_Vesely_106_SIMULATED.csv.
<br>The dataset assuming a homogeneous, uniform, and time-invariant mortality rate across age groups (at about real world level). Afterward Individuals were randomly assigned to vaccinated or unvaccinated cohorts, with real-world dosing schedules applied. Enforcing that death could only occur post-vaccination (real world) inherently introducing immortal time bias, as illustrated below.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/F)%20rolling%20daily%20crude%20HR%20by%20age/F%29%20sim%20MINBIAS%20rolling%20daily%20crude%20HR%20by%20age.png width="1280" height="auto">
<br>
_________________________________________
### J) Bias Study ratio vx_uvx.py

<br>Phyton script [J) Bias Study ratio vx_uvx.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/J%29%20Bias%20Study%20ratio%20vx_uvx.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/J%29%20Bias%20study%20ratio%20vx%20uvx/J%29%20Bias%20study%20ratio%20over%20time%20%20TimeDependend.html)
<br>
<br>This script simulates uniform death risk over time to test bias in survival analysis. It compares static vs. time-dependent vaccinated/unvaccinated classification, computes death rates and 1st derivatives, Kaplan-Meier curves, and Cox models, and visualizes the results in an interactive Plotly HTML plot.
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/J)%20Bias%20study%20ratio%20vx%20uvx/J)%20Bias%20study%20ratio%20over%20time%20%20TimeDependend.png width="1280" height="auto">
<br>
_________________________________________

### ZB) Hypergeometric (used by Charles Sanders Peirce) Vaccine Effectiveness Analysis with Confidence Intervals

<br>Phyton script [ZB) CS-Pierce Hypergeometric VaxCodes.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/ZB%29%20CS-Pierce%20Hypergeometric%20VaxCodes.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZB%29%20CS-Pierce%20Hypergeometric%20VaxCodes/ZB%29%20hypergeom_vaccine_effectiveness_with_CI_and_vaxcode_age.html)
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZB%29%20CS-Pierce%20Hypergeometric%20VaxCodes/ZB)%20hypergeom_vaccine_effectiveness_with_CI_and_vaxcode_age.png width="1280" height="auto">
<br>
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZB%29%20CS-Pierce%20Hypergeometric%20VaxCodes/ZB%29%20sim%20hypergeom_vaccine_effectiveness_with_CI_and_vaxcode_age.png width="1280" height="auto">
<br>
_________________________________________

### ZC) 

<br>Phyton script [ZC) dowhy vaxcode doses vs total_death individual.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/ZC%29%20dowhy%20vaxcode%20doses%20vs%20total_death%20individual.py) 
<br> [Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZC%29%20dowhy%20vaxcode%20doses%20vs%20totaldeath%20individual/ZC%29%20dowhy_vaxcode_doses_vs_deaths.html)
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZC%29%20dowhy%20vaxcode%20doses%20vs%20totaldeath%20individual/ZC%29%20dowhy_vaxcode_doses_vs_deaths.png width="1280" height="auto">
<br>
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZC)%20dowhy%20vaxcode%20doses%20vs%20totaldeath%20individual/ZC)%20sim%20dowhy_vaxcode_doses_vs_deaths.png width="1280" height="auto">
<br>
_________________________________________

### More plots added:
_________________________________________
### AC) Mean age at death before and after the start of vaccination czech real world data

<br>Phyton script [AC) age_mean.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/A%29%20age_mean.py) 
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/AC)%20age%20mean%20pop/AC)%20age_mean_pop.png width="1280" height="auto">
<br>
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/AC)%20age%20mean%20pop/AC)%20sim%20no%20bias%20age_mean_pop.png width="1280" height="auto">
<br>
_________________________________________
### EA) Days since Doses

<br>Phyton script [EA) batches vs death.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/EA%29%20batches%20vs%20death.py) 
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/EA)%20batches%20vs%20death/hist_chunks/batch_hist_1_to_100.png width="1280" height="auto">
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/EA)%20batches%20vs%20death/hist_chunks/batch_hist_201_to_300.png width="1280" height="auto">
<br>
_________________________________________
### UA) 

<br>Phyton script [UA) diff death dose agebin.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/UA%29%20diff%20death%20dose%20agebin.py) 
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/UA)%20diff%20death%20dose%20agebin/rolling_corr_doses_vs_diff.png width="1280" height="auto">
<br>
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/UA%29%20diff%20death%20dose%20agebin/sim%20no%20bias%20rolling_corr_doses_vs_diff.png width="1280" height="auto"> 
<br>
_________________________________________

### UC) 
<br>**Rolling correlation between the daily vaccine dose curve and the difference in normalized death rates between uvx and vx individuals (uvx - vx) !! The difference in deaths (uvx - vx) compensates for external influences, so vax effect should be left!!**

<br>Phyton script [UC) diff norm death dose agebin.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/UC%29%20diff%20norm%20death%20dose%20agebin.py) 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/UC%29%20diff%20norm%20death%20dose%20agebin/norm%20rolling_corr_doses_vs_diff.png width="1280" height="auto">
<br>

[Download interactive html](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/UC%29%20diff%20norm%20death%20dose%20agebin/norm%20rolling_corr_doses_vs_diff.html)

<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br**>As the simulated Death risk for the whole homogen population is constant over time, differnece uvx-vx should fluctuate horizontally around level 0!** 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/UC)%20diff%20norm%20death%20dose%20agebin/sim%20no%20bias%20norm%20rolling_corr_doses_vs_diff.png width="1280" height="auto"> 
<br>
_________________________________________

### ZG) 

<br>Phyton script [ZG) doses_vs_deaths_dowhy.png](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/ZG%29%20dowhy%20doses%20vs%20death.py) 
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZG)%20dowhy%20doses%20vs%20death/ZG%29%20doses_vs_deaths_dowhy.png width="1280" height="auto">
<br>
<br>Zoomd in
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZG)%20dowhy%20doses%20vs%20death/ZG%29%20doses_vs_deaths_dowhy_zoom.png width="1280" height="auto">
<br>
<br>**DoWhy seems not to be correct here**
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br> 
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZG)%20dowhy%20doses%20vs%20death/ZG%29%20sim%20no%20bias%20doses_vs_deaths_dowhy.png width="1280" height="auto"> 
<br>
<br>Zoomd in should be horizonta line (same simulated death reate for all three traces)
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/ZG)%20dowhy%20doses%20vs%20death/ZG%29%20sim%20no%20bias%20doses_vs_deaths_dowhy_zoom.png width="1280" height="auto"> 
<br>
_________________________________________
### S) 

<br>Phyton script [S) diff death dose agebin.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/S%29%20diff%20death%20dose%20agebin.py) 
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/S%29%20diff%20death%20dose%20agebin/S)%20vx%20uvx%20raw%20diff%20population%20doses%20causal%20estimate.png width="1280" height="auto">
<br>
<br>Zoomd in
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/S)%20diff%20death%20dose%20agebin/S%29%20vx%20uvx%20raw%20diff%20population%20doses%20causal%20estimate_zoom.png width="1280" height="auto">
<br>
<br>**DoWhy seems not to be correct here**
<br>To test for bias, I run the same code on simulated data with a uniform, constant death rate across ages and time. I then **afterwards** split into vaccinated and unvaccinated groups, ignoring real-world constraints like requiring death to occur after vaccination, which would introduce selection bias.
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/S)%20diff%20death%20dose%20agebin/S%29%20sim%20no%20bias%20vx%20uvx%20raw%20diff%20population%20doses%20causal%20estimate.png width="1280" height="auto"> 
<br>
_________________________________________
### UB) 

<br>Phyton script [UB) diff death dose agebin.py](https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Py%20Scripts/UB%29%20diff%20death%20dose%20agebin.py)
<br>
<br>
<img src=https://github.com/gitfrid/CzechFOI-DRATE/blob/main/Plot%20Results/UB%29%20diff%20death%20dose%20agebin/dowhy_scatter_doses_vs_diff.png width="1280" height="auto">
<br>
_________________________________________
### Software Requirements:

These scripts don't require SQLite queries to aggregate the 11 million individual data rows.
Instead, the aggregation is handled directly by Python scripts, which can generate aggregated CSV files very quickly.
For coding questions or help, visit https://chatgpt.com.

- [Python 3.12.5](https://www.python.org/downloads/) to run the scripts.
- [Visual Studio Code 1.92.2](https://code.visualstudio.com/download) to edit and run scripts.


### Disclaimer:
**The results have not been checked for errors. Neither methodological nor technical checks or data cleansing have been performed.**


