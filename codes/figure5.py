import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

osi_osm_completeness = {'1:1': 35.24416135881104, '1:0': 1.910828025477707, '1:many': 12.845010615711253, 'many:1': 36.624203821656046, 'many:many': 12.420382165605096}

osm_ai_completeness = {'1:1': 28.874734607218684, '1:0': 0.0, '1:many': 21.125265392781316, 'many:1': 32.59023354564756, 'many:many': 17.40976645435244}

osi_ai_completeness = {'1:1': 36.69250645994832, '1:0': 0.0, '1:many': 11.83749354005168, 'many:1': 42.91416135881104, 'many:many': 8.57583864118896}


labels = list(osi_osm_completeness.keys())
values = list(osi_osm_completeness.values())#[completeness['1:1'], completeness['1:0'], completeness['1:many'], completeness['many:1'], completeness['many:many']]
values2 = list(osi_ai_completeness.values())
values3 = list(osm_ai_completeness.values())

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
plt.figure(figsize=(10, 5))
# change the color of the bar


plt.bar(x, values, width, label='OSi vs OSM', color=(0.2, 0.4, 0.6, 0.6))
plt.bar(x + width, values2, width, label='OSi vs OSi-GAN', color=(0.2, 0.4, 0.6, 0.4))
plt.bar(x + 2*width, values3, width, label='OSi vs OSM-GAN', color=(0.2, 0.4, 0.6, 0.2))

plt.xticks(x+width, labels)
# use percentage to 100 
plt.yticks(np.arange(0, 101, 10))
plt.ylabel('Completeness (%)')
plt.xlabel('Relationship Categories')
plt.title('Completeness Comparison')
plt.legend()
plt.show()