select
    fold,
    hotel_id,
    origin_date,
    stay_date,
    horizon,
    actual_rooms,
    forecast_rooms,
    p10_rooms,
    p90_rooms,
    rms as rms_forecast,
    naive_y1
from {{ source('raw', 'backtest') }}
