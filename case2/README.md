Data and scripts for Case 2

Steps to follow to reproduce PINN results

- Run parser.py in data to generate dataset. The file TecPlot2.dat can be
  downloaded from: <insert link once ready>.
- Download turbmat (https://github.com/idies/turbmat) and place it in the dns
  directory. Run fetchDNS.m in dns to retrieve original data from the Johns
  Hopkins Turbulence Database.
- Run run_pinn.py to process data. As in the manuscript, the first half of the
  dataset can be processed by setting "t0 = 28" in param.py, while the second
  dataset can be analyzed by setting "t0 = 28 + 970".
