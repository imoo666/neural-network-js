import { HashRouter, Routes } from "react-router-dom";
import { renderRoute, routes } from "./config/routes";

export default function App() {
  return (
    <HashRouter>
      <Routes>{renderRoute(routes)}</Routes>
    </HashRouter>
  );
}
