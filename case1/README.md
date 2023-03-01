Data and scripts for Case 1

Steps to follow to reproduce PINN results

- Download and untar in cases data_cases.tgz from
  https://zenodo.org/record/7688953#.Y_92tNLMKV4. As explained in the
  manuscript, each case is characterized by the number of seeded particles and
  the added errors. So for example, case Np_04000_Er_0 uses 4000 particles and
  has no added errors. Each case has 40 different realizations which were used
  to ensemble average the results.
- Download turbmat (https://github.com/idies/turbmat) and place it in the dns
  directory. Run fetchDNS.m in dns to retrieve original data from the Johns
  Hopkins Turbulence Database.
- Copy all \*.py files to each of the subdirectories in cases, e.g.
  cases/Np_04000_Er_0/01, to run each case.
