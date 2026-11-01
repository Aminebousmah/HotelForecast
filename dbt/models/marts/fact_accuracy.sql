with forecasts as (
    select
        f.horizon,
        h.brand,
        h.region,
        f.actual_rooms,
        f.forecast_rooms,
        f.rms_forecast,
        f.naive_y1
    from {{ ref('fact_forecast') }} as f
    inner join {{ ref('dim_hotel') }} as h on f.hotel_id = h.hotel_id
)

select
    horizon,
    brand,
    region,
    sum(abs(actual_rooms - forecast_rooms)) / sum(actual_rooms) as wape_model,
    sum(abs(actual_rooms - rms_forecast)) / sum(actual_rooms) as wape_rms,
    sum(abs(actual_rooms - naive_y1)) / sum(actual_rooms) as wape_naive,
    sum(forecast_rooms - actual_rooms) / sum(actual_rooms) as bias_model
from forecasts
group by horizon, brand, region
