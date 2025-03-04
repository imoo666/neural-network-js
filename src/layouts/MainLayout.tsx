import { Layout, Menu } from "antd";
import { useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { renderMenuItems, routes } from "../config/routes";

const { Sider, Content } = Layout;

export default function MainLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const menu = routes[0].children || [];

  return (
    <Layout className="min-h-screen">
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        className="fixed h-screen left-0 top-0 bottom-0 z-20"
      >
        <div className="h-8 bg-gray-700 m-4 rounded" />
        <Menu
          theme="dark"
          selectedKeys={[location.pathname.slice(1)]}
          mode="inline"
          items={renderMenuItems(menu)}
          onClick={(e) => {
            navigate(e.key);
          }}
        />
      </Sider>
      <Layout>
        <Content className="p-6 h-[calc(100vh-64px)] overflow-y-auto">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
