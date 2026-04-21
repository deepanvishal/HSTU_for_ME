WITH florida_boundary AS (
  SELECT state_geom
  FROM `bigquery-public-data.geo_us_boundaries.states`
  WHERE state_fips_code = '12'
)

SELECT 
  z.zip_code,
  z.area_land_meters / 2589988.11                    AS area_sq_miles,
  ST_CENTROID(z.zip_code_geom)                       AS zip_centroid,
  ST_Y(ST_CENTROID(z.zip_code_geom))                 AS zip_lat,
  ST_X(ST_CENTROID(z.zip_code_geom))                 AS zip_long
FROM `bigquery-public-data.geo_us_boundaries.zip_codes` z
CROSS JOIN florida_boundary f
WHERE ST_INTERSECTS(z.zip_code_geom, f.state_geom)
