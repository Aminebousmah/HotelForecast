select
    hotel_id,
    stay_date,
    capacity,
    rooms_sold,
    rooms_occupied,
    rooms_denied,
    room_revenue,
    rooms_sold / capacity as occupancy,
    if(rooms_sold > 0, room_revenue / rooms_sold, null) as adr,
    room_revenue / capacity as revpar
from {{ ref('stg_daily_hotel') }}
