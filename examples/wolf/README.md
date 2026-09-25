# Wolf example assets

Only the lightweight wolf_scene.usda scene entry point and wolf_transform.json are versioned in this repository.

To run the demo, provide these files in this directory:

~~~text
wolf_crop.ply
wolf_graph.npz
wolf_splat.usd
~~~

All three must come from the same prepared 3DGS asset. wolf_scene.usda references wolf_splat.usd, while the runtime loads wolf_crop.ply and wolf_graph.npz in the same Gaussian ordering.

Do not publish these generated assets until the original dataset and redistribution terms are documented in docs/data.md.