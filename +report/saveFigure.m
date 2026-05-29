function saveFigure(fig, stem)
% saveFigure  Write a figure to <stem>.png and <stem>.pdf (no-op if stem empty).
    if isempty(stem); return; end
    d = fileparts(stem);
    if ~isempty(d) && ~isfolder(d); mkdir(d); end
    exportgraphics(fig, [stem '.png'], 'Resolution', 150);
    exportgraphics(fig, [stem '.pdf'], 'ContentType', 'vector');
end
