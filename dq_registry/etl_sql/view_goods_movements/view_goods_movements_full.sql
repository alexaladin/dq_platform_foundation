select g.movement_id
       ,g.material_id
       ,m.material_name
       ,m.weight * g.quantity as total_product_weight
       ,g.quantity
       ,g.movement_type
       ,g.plant
       ,g.movement_date
       ,g.ts_load
       ,g.forecast_date
  from raw.goods_movement g
       left join raw.material_maseter m on g.material_id = m.material_id
 where movement_type = 'TRANSFER'
