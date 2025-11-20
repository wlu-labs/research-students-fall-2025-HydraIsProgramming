-- Load Ontario EV registrations (already staged and cleaned CSV)
TRUNCATE staging.vehicle_regs_min_raw;

COPY staging.vehicle_regs_min_raw (ref_date, geo, fuel_type, value)
FROM 'C:/EV_Project/Data/2010002501_databaseLoadingData.csv'
DELIMITER ','
CSV HEADER;


TRUNCATE core.vehicle_registrations;
-- Transform into clean table
INSERT INTO core.vehicle_registrations (ref_date, geo, fuel_type, vehicles)
SELECT
  TO_DATE(ref_date, 'Mon-YY') AS ref_date,
  geo,
  fuel_type,
  value
FROM staging.vehicle_regs_min_raw
WHERE geo ILIKE 'Ontario%'
  AND fuel_type IN ('Battery electric','Plug-in hybrid electric')
  AND value IS NOT NULL
ON CONFLICT DO NOTHING;  -- prevents duplicate insertions

TRUNCATE TABLE core.toronto_neighbourhood_stats;

SELECT * 
FROM core.vehicle_registrations
ORDER BY ref_date DESC

COPY core.toronto_neighbourhood_stats (hdnum, hdname, population_total, median_household_income_2020)
FROM 'C:/EV_Project/Data/toronto_neighbourhood_population_income_2021.csv'
DELIMITER ','
CSV HEADER;

TRUNCATE ext.afdc_stations_json;
COPY ext.afdc_stations_rawtext(line)
FROM 'C:\EV_Project\Data\afdc_ev_toronto.json' WITH (FORMAT text);
