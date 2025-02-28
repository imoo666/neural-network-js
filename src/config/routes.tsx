import { HomeOutlined, SettingOutlined, UserOutlined } from "@ant-design/icons";
import { GetProp, Menu, MenuProps } from "antd";
import { ReactNode } from "react";
import { Outlet, Route, RouteObject } from "react-router-dom";
import MainLayout from "../layouts/MainLayout";
import Home from "../pages/home/Index";
import Account from "../pages/settings/account";
import Profile from "../pages/settings/profile";

interface Menu {
  title: string;
  icon?: ReactNode;
  hidden?: boolean;
}
export type RouteItem = {
  path: string;
  children?: RouteItem[];
  replace?: string;
  menu?: Menu;
} & RouteObject;

export const routes: RouteItem[] = [
  {
    path: "/",
    element: <MainLayout />,
    children: [
      {
        path: "home",
        element: <Home />,
        menu: { title: "主页", icon: <HomeOutlined /> },
      },
      {
        path: "settings",
        element: <Outlet />,
        menu: { title: "设置", icon: <SettingOutlined /> },
        children: [
          {
            path: "account",
            element: <Account />,
            menu: { title: "账户设置", icon: <UserOutlined /> },
          },
          {
            path: "profile",
            element: <Profile />,
            menu: { title: "个人资料", icon: <UserOutlined /> },
          },
        ],
      },
    ],
  },
  {
    path: "/login",
    element: <div>登录</div>,
  },
];

export const renderRoute = (
  routes: RouteItem[],
  parentPath: string = ""
): ReactNode => {
  return (
    <>
      {routes.map((item) => {
        const { path, element, children } = item;
        const fullPath = parentPath ? `${parentPath}/${path}` : path;
        return (
          <Route key={fullPath} path={path} element={element}>
            {children && renderRoute(children, path)}
          </Route>
        );
      })}
    </>
  );
};

// 将我们的路由转换为适配antd菜单的格式
type MenuItem = GetProp<MenuProps, "items">[number];
export const renderMenuItems = (
  items: RouteItem[],
  parentPath = ""
): MenuItem[] => {
  const result = items
    .filter((item) => item.menu && !item.menu?.hidden) // 过滤隐藏菜单项
    .map((item) => {
      const { menu, path, children } = item;
      const fullPath = parentPath ? `${parentPath}/${path}` : path;
      const result: MenuItem = {
        key: fullPath,
        label: menu?.title,
        icon: menu?.icon,
        children: children ? renderMenuItems(children, fullPath) : undefined,
      };
      return result;
    });
  return result;
};
