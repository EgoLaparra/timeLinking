from lxml import etree


def get_schema():

    #tnschema = etree.parse('/Users/laparra/Data/Datasets/Time/SCATE/anafora-annotations/.schema/timenorm-schema.xml')
    tnschema = etree.parse('/home/egoitz/Data/Datasets/Time/SCATE/anafora-annotations/.schema/timenorm-schema.xml')
    schema = dict()
    for entity in tnschema.findall('.//entity'):
        type = entity.get('type')
        for property in entity.findall('.//property'):
            ptype = property.get('type')
            required = bool(property.get('required'))
            if required is None:
                required = "True"
            instacesOf = property.get('instanceOf')
            if instacesOf is None:
                instacesOf = []
            else:
                instacesOf = instacesOf.split(',')
            if type not in schema:
                schema[type] = dict()
            schema[type][ptype] = (required,instacesOf)

    return schema