# MICO 

This repository contains the code of the state of the art MICO, a probabilistic 
optimization tool for dynamic control of construction projects. Project management is 
becoming increasingly complex due to a shift of contractual responsibilities to 
contractors, broader project scopes with associated increase in interfaces, and the 
increasing influence of (local) stakeholders. This calls for adaptive decision support 
both for the plan and execution phase in order to find the best fitting solution for 
multiple (and sometimes changing) objectives. For this aim, the MICO has been developed. 
It is based upon the MitC concept ([Kammouh et al., 2021](https://doi.org/10.1061/(ASCE)CO.1943-7862.0002126); 
[Kammouh et al., 2022](https://doi.org/10.1016/j.autcon.2022.104450)). 
It integrates preference based IMAP optimization and probabilistic network planning, 
incorporating both activity uncertainties and risk events.

The IMAP optimization is based on [the Preferendus](https://github.com/TUDelft-Odesys/Preferendus) 
principles following the Odesys methodology ([Wolfert, 2023](https://doi.org/10.3233/RIDS10); 
[Van Heukelum et al., 2024](https://doi.org/10.1080/15732479.2023.2297891)). 
For preference aggregation as part of IMAP, the [A-fine Aggregator](https://github.com/Boskalis-python/a-fine-aggregator) algorithm 
is used.

The MICO was a co-development of Niels Roeders, Lukas Teuber, Harold van Heukelum and 
Rogier Wolfert at Boskalis R&D.

## License

This repository is licensed under the [MIT license](https://choosealicense.com/licenses/mit/).
