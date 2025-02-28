# React + TypeScript + Vite + antdesign

这是一个配置好的起始框架

- "vite": "^6.0.5"
- "react-router-dom": "^7.2.0"
- "tailwindcss": "^4.0.9"
- "react": "^18.3.1"
- "antd": "^5.24.2"

拥有一个优雅的路由系统

```js
interface Menu {
  title: string;
  icon?: ReactNode;
  hidden?: boolean;
}
export type RouteItem = {
  path: string,
  children?: RouteItem[],
  replace?: string,
  menu?: Menu,
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
```

只需要配置这个 routes，即可同时生成路由和菜单
![image](https://github.com/user-attachments/assets/009dc385-3927-4a2f-afef-5b6c4c312fb7)
