select
    hotel_id,
    stay_date,
    capacity,
    rooms_sold,
    rooms_occupied,
    rooms_denied,
    room_revenue
from {{ source('raw', 'daily_hotel') }}
