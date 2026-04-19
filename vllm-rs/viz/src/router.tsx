import {
  Outlet,
  RootRoute,
  Route,
  Router,
  createHashHistory,
} from "@tanstack/react-router";
import { GalleryRoute } from "./routes/Gallery";
import { ZoomRoute } from "./routes/Zoom";
import { DiffRoute } from "./routes/Diff";
import { Shell } from "./routes/Shell";

// All routes share a hash-history root so the URL never depends on
// server-side rewrites — the SPA ships as static assets and survives
// being dropped behind any HTTP server.
const hashHistory = createHashHistory();

// `sel` rides as a search-param on the root so the cross-arch
// local-flow highlight persists across navigation (gallery ↔ zoom ↔
// diff) and is shareable via URL. View toggling between math / source
// / fuf is now drilldown-only — the gallery is fixed on math.
//
// Hand-validated to avoid pulling in Zod. Bad values fall back to
// defaults rather than throwing — the URL is user-editable.
export interface RootSearch {
  sel: string | null;
}
function validateSearch(search: Record<string, unknown>): RootSearch {
  const sel = typeof search.sel === "string" && search.sel ? search.sel : null;
  return { sel };
}

const rootRoute = new RootRoute({
  component: () => (
    <Shell>
      <Outlet />
    </Shell>
  ),
  validateSearch,
});

const galleryRoute = new Route({
  getParentRoute: () => rootRoute,
  path: "/",
  component: GalleryRoute,
});

const zoomRoute = new Route({
  getParentRoute: () => rootRoute,
  path: "zoom/$arch",
  component: ZoomRoute,
});

const diffRoute = new Route({
  getParentRoute: () => rootRoute,
  path: "diff/$a/$b",
  component: DiffRoute,
});

const routeTree = rootRoute.addChildren([galleryRoute, zoomRoute, diffRoute]);

export const router = new Router({
  routeTree,
  history: hashHistory,
  defaultPreload: false,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
