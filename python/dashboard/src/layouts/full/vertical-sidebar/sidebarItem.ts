export interface menu {
  header?: string;
  title?: string;
  icon?: string;
  to?: string;
  divider?: boolean;
  chip?: string;
  chipColor?: string;
  chipVariant?: string;
  chipIcon?: string;
  children?: menu[];
  disabled?: boolean;
  type?: string;
  subCaption?: string;
}

const sidebarItem: menu[] = [
  {
    title: '统计',
    icon: 'mdi-view-dashboard',
    to: '/dashboard/default'
  },
  {
    title: '配置文件',
    icon: 'mdi-cog',
    to: '/config',
  },
  {
    title: '插件',
    icon: 'mdi-puzzle',
    to: '/extension'
  },
  {
    title: '聊天',
    icon: 'mdi-chat',
    to: '/chat'
  },
  {
    title: '控制台',
    icon: 'mdi-console',
    to: '/console'
  },
  {
    title: '设置',
    icon: 'mdi-wrench',
    to: '/settings'
  },
  {
    title: '关于',
    icon: 'mdi-information',
    to: '/about'
  },
  // {
  //   title: 'Project ATRI',
  //   icon: 'mdi-grain',
  //   to: '/project-atri'
  // },
];

export default sidebarItem;
