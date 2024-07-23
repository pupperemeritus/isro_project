# Slant to Vertical Conversion

## Phase scintillation data

```python
c = filtered_data['Phi60 on Sig1, 60-second phase sigma (radians)'].values
```

## Amplitude scintillation data

```python
b = filtered_data['Total S4 on Sig1 (dimensionless)'].values
```

## amplitude conversion

```python
RE = 6378
H_ipp = 350
p = pd.to_numeric(filtered_data['p on Sig1, spectral slope of detrended phase in the 0.1 to 25Hz range (dimensionless)'], errors='coerce').values
T1 = RE _np.cos(e) / (RE + H_ipp)
T2 = np.sqrt(1 - T1_*2)
T3 = (1/T2)**(p+0.25)
Vert_A = b / T3
```

## phase conversion

```python
RE = 6378
H_ipp = 350
p = pd.to_numeric(filtered_data['p on Sig1, spectral slope of detrended phase in the 0.1 to 25Hz range (dimensionless)'], errors='coerce').values
T1 = RE _np.cos(e) / (RE + H_ipp)
T2 = np.sqrt(1 - T1_*2)
T3 = (1/T2)**0.5
Vert_A = c / T3
```
