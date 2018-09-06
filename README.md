# Bilayer Graphene in Tunneling Measurements

Bilayer graphene (BLG) is of great technological interest due to its dynamic band structure. By applying an electric field, a band gap opens changing bilayer graphene from a semimetal to a semiconductor. One method of probing the band structure of bilayer graphene is through tunneling measurements such as electron tunneling spectroscopy and scanning tunneling spectroscopy. However, when applying voltages to the gates in a tunneling experiment, an electric field is produced which changes the magnitude of the gap and the shape of the band structure in general. Therefore, the current from bilayer graphene is much more complicated than one may expect.

<img src=images/BandStructure/GatedBLG.png width="800">

This package uses a parallel plate capacitor model to analyze bilayer graphene in tunneling measurements. It is applicable even to STM measurements since the radius of curvature of an STM tip is much larger than its distance from the surface. The package includes methods for calculating common quantities associated with solid state systems (band structure, density of states, etc.) as well as methods for analyzing the signal of bilayer graphene in a solid state system.

A series of Jupyter notebooks is available for a tutorial in the directory 'Notebooks'. They cover both the theory of bilayer graphene's band structure as well as the code which uses the theory for performing calculations.