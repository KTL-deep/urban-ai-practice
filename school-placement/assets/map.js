window.dashExtensions = window.dashExtensions || {};
window.dashExtensions.schoolPlacement = {
  map: {
    style: function(feature, context) {
      const selected = feature.properties.block_id === context.hideout.selected;
      return {
        color: selected ? '#122334' : '#f9fcff',
        weight: selected ? 4 : 1,
        fillColor: feature.properties.map_color || '#d5dce4',
        fillOpacity: 0.85
      };
    }
  }
};
// dash-leaflet 1.1.x resolves Namespace variables directly on window.
window.schoolPlacement = window.dashExtensions.schoolPlacement;
